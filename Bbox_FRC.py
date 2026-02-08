import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

class CocoOneClassDetection(Dataset):
    def __init__(self, images_dir: str, ann_path: str):
        self.images_dir = images_dir

        with open(ann_path, "r") as f:
            coco = json.load(f)

        # COCO: images list, annotations list
        self.images = coco["images"]
        self.ann = coco.get("annotations", [])

        # Index annotations by image_id for fast lookup
        self.ann_by_image = {}
        for a in self.ann:
            img_id = a["image_id"]
            self.ann_by_image.setdefault(img_id, []).append(a)

        # Map dataset index -> COCO image entry
        self.id_to_img = {img["id"]: img for img in self.images}
        self.ids = [img["id"] for img in self.images]  # stable ordering

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.id_to_img[img_id]

        # Roboflow COCO usually uses "file_name"
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)  # float32 [0..1], shape [3,H,W]

        anns = self.ann_by_image.get(img_id, [])

        boxes = []
        for a in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue  # skip tiny/degenerate boxes
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            # ONE CLASS: label = 1 for every box
            labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CocoOneClassDetection(
    images_dir="path/to/images",
    ann_path="path/to/annotations.json"
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2,
                    collate_fn=collate_fn)


def make_model():
    num_classes = 2  # background(0) + fuel/ball(1)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(10):
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch} loss {loss.item():.4f}")

import torch

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: [N,4]  (x1,y1,x2,y2)
    boxes2: [M,4]
    returns IoU matrix: [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)                               # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]                         # [N,M]

    # Union
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def match_detections_to_gt(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """
    Greedy matching: sort preds by score desc, match each pred to best unmatched GT by IoU.
    Returns TP, FP, FN counts for this one image.
    """
    device = pred_boxes.device
    gt_boxes = gt_boxes.to(device)

    # Edge cases
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 0, 0, 0
    if pred_boxes.numel() == 0:
        return 0, 0, gt_boxes.shape[0]
    if gt_boxes.numel() == 0:
        return 0, pred_boxes.shape[0], 0

    # Sort predictions by confidence
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    ious = box_iou(pred_boxes, gt_boxes)  # [P,G]
    gt_matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=device)

    tp = 0
    fp = 0

    for p in range(pred_boxes.shape[0]):
        # best GT for this pred
        best_iou, best_g = torch.max(ious[p], dim=0)
        if best_iou >= iou_thresh and not gt_matched[best_g]:
            tp += 1
            gt_matched[best_g] = True
        else:
            fp += 1

    fn = int((~gt_matched).sum().item())
    return tp, fp, fn


@torch.no_grad()
def validate_detector(model, val_loader, device, score_thresh=0.5, iou_thresh=0.5, max_batches=None):
    """
    Computes dataset-level Precision/Recall/F1 using IoU matching.

    Assumes targets are TorchVision format:
      target["boxes"] [N,4]
      target["labels"] [N] (ignored here since you are 1-class)
    """
    model.eval()

    total_tp = total_fp = total_fn = 0
    seen = 0

    for b, (images, targets) in enumerate(val_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)  # list of dicts

        for pred, tgt in zip(preds, targets):
            # keep only confident predictions
            scores = pred["scores"]
            keep = scores >= score_thresh

            pred_boxes = pred["boxes"][keep]
            pred_scores = scores[keep]

            gt_boxes = tgt["boxes"]

            tp, fp, fn = match_detections_to_gt(pred_boxes, pred_scores, gt_boxes, iou_thresh=iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            seen += 1

        if max_batches is not None and (b + 1) >= max_batches:
            break

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "images_evaluated": seen,
        "score_thresh": score_thresh,
        "iou_thresh": iou_thresh,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    print(f"Using device '{device}'")
    print("\nBeginning training...")
    model = make_model().to(device)
    train(model)


if __name__ == "__main__":
    main()
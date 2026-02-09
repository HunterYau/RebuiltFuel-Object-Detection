import json
import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ColorJitter

CHECKPOINT_PATH = "./FRC 2026 Fuel.v2i.coco/model.pth"


class CocoOneClassDetection(Dataset):
    def __init__(
        self,
        images_dir: str,
        ann_path: str,
        augment: bool = False,
        hflip_p: float = 0.5,
        contrast_jitter_p: float = 0.8,
        contrast_range=(0.6, 1.4),
    ):
        self.images_dir = images_dir
        self.augment = augment

        self.hflip_p = hflip_p
        self.contrast_jitter_p = contrast_jitter_p

        # Contrast-only jitter (works for grayscale too)
        self.contrast_jitter = ColorJitter(
            contrast=(contrast_range[0], contrast_range[1])
        )

        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.ann = coco.get("annotations", [])

        # image_id -> annotations
        self.ann_by_image = {}
        for a in self.ann:
            img_id = a["image_id"]
            self.ann_by_image.setdefault(img_id, []).append(a)

        self.id_to_img = {img["id"]: img for img in self.images}
        self.ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.id_to_img[img_id]

        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")  # keep RGB for TorchVision
        img = F.to_tensor(img)  # [3,H,W], float in [0,1]

        anns = self.ann_by_image.get(img_id, [])

        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]  # COCO format
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)

        # -------- AUGMENTATIONS --------
        if self.augment:
            # contrast jitter (image only)
            if random.random() < self.contrast_jitter_p:
                img = self.contrast_jitter(img)

            # horizontal flip (image + boxes)
            if random.random() < self.hflip_p:
                _, _, W = img.shape
                img = F.hflip(img)

                if boxes_t.numel() > 0:
                    x1 = boxes_t[:, 0].clone()
                    x2 = boxes_t[:, 2].clone()
                    boxes_t[:, 0] = W - x2
                    boxes_t[:, 2] = W - x1

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = CocoOneClassDetection(
    images_dir="./FRC 2026 Fuel.v2i.coco/train/images",
    ann_path="./FRC 2026 Fuel.v2i.coco/train/_annotations.coco.json",
    augment=True,
    hflip_p=0.5,
    contrast_jitter_p=0.8,
    contrast_range=(0.7, 1.3),
)

val_dataset = CocoOneClassDetection(
    images_dir="./FRC 2026 Fuel.v2i.coco/valid/images",
    ann_path="./FRC 2026 Fuel.v2i.coco/valid/_annotations.coco.json",
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=7, persistent_workers=True,
                    collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, batch_size=3, shuffle=True, num_workers=7, persistent_workers=True,
                    collate_fn=collate_fn)


def make_model():
    num_classes = 2  # background(0) + fuel/ball(1)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(model, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler(device)

    total_loss = 0
    for epoch in range(epochs):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with autocast(device_type=device):
                loss_dict = model(images, targets)

            loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        metrics = validate_detector(model, val_loader, device)

        print(f"epoch {epoch+1}, loss {total_loss/len(train_loader):.4f}, Correct Detections: {metrics["TP"]}, "
              f"Hallucinations: {metrics["FP"]}, Missed Objects: {metrics["FN"]}, Precision: {metrics["precision"]}, "
              f"Correct Objects: {metrics["recall"]}, Balance: {metrics["f1"]}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            CHECKPOINT_PATH
        )


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


def _to_uint8_img(img_chw: torch.Tensor) -> torch.Tensor:
    """
    img_chw: float tensor [3,H,W] in 0..1 (typical from torchvision transforms)
    returns uint8 [3,H,W]
    """
    img = img_chw.detach().cpu()
    img = (img.clamp(0, 1) * 255).to(torch.uint8)
    return img

@torch.no_grad()
def visualize_predictions(
    model,
    dataset=None,
    img=None,
    target=None,
    idx=0,
    device="cuda",
    score_thresh=0.5,
    show_gt=True,
    max_boxes=50,
):
    """
    Visualize predicted boxes (red) and optional GT boxes (green).

    Use either:
      - dataset + idx, OR
      - img (+ optional target)

    dataset should return (img, target) like TorchVision detection datasets:
      img: [3,H,W] float
      target: {"boxes":[N,4], ...}
    """
    assert (dataset is not None) or (img is not None), "Provide dataset or img."

    model.eval()

    if img is None:
        img, target = dataset[idx]

    img_in = img.to(device)
    pred = model([img_in])[0]

    boxes = pred["boxes"].detach().cpu()
    scores = pred["scores"].detach().cpu()

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    # limit clutter
    if boxes.shape[0] > max_boxes:
        boxes = boxes[:max_boxes]
        scores = scores[:max_boxes]

    # Prepare image for matplotlib: HWC in 0..1
    img_uint8 = _to_uint8_img(img)
    img_hwc = img_uint8.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(img_hwc)
    ax.set_axis_off()

    # GT boxes (green)
    if show_gt and (target is not None) and ("boxes" in target):
        gt_boxes = target["boxes"].detach().cpu()
        for b in gt_boxes:
            x1, y1, x2, y2 = b.tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

    # Pred boxes (red) + score label
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = b.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 6, 0), f"{s:.2f}", color="red", fontsize=10,
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    title = f"Pred (red) @ score≥{score_thresh}"
    if show_gt and (target is not None):
        title += " | GT (green)"
    ax.set_title(title)
    plt.show()

    return pred  # in case you want the raw outputs


def main():
    print(f"Using device '{device}'")

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Checkpoint loaded.")

    print("\nBeginning training...")
    # train(model, 5)

    pred = visualize_predictions(model, dataset=val_dataset, idx=0, device=device, score_thresh=0.4, show_gt=False)


if __name__ == "__main__":
    main()
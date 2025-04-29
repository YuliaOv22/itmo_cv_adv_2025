#!/usr/bin/env python3
# infer_yolov1.py
import argparse
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

from model_scripts.YOLOv1 import YoloV1    
from utils.utils import cellboxes_to_boxes             

import torchvision.models as models

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

def load_model(weights: Path, device: str = "cuda"):
    mobilenet = models.mobilenet_v2(weights=None)
    backbone = mobilenet.features
    model = YoloV1(
        backbone=backbone,
        split_size=7,
        num_boxes=2,
        num_classes=3,
        backbone_out_features=1280
    ).to(device)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

@torch.no_grad()
def predict(model, image_path: Path, device: str = "cuda",
            img_size: int = 448, conf_thres: float = 0.4, iou_thres: float = 0.5):
    """
    Возвращает тензор изображения (3 × H × W) и список боксов
    `[[class, conf, x, y, w, h], …]` в формате midpoint, нормированный [0,1].
    """
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    preds = model(img)              # (1, S, S, 30)
    bboxes = cellboxes_to_boxes(preds)[0]  # уже отфильтрованные NMS-ом

    # ещё раз порогconfidence, если нужно
    bboxes = [b for b in bboxes if b[1] >= conf_thres]

    return img.squeeze(0).cpu(), bboxes

def draw_bboxes(img_tensor, bboxes, save_to: Path | None = None):
    """
    Рисует bbox-ы. Цвет зависит от класса.
    """
    img = img_tensor.permute(1, 2, 0).numpy()
    H, W = img.shape[:2]

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")

    for cls, conf, x, y, w, h in bboxes:
        x1 = (x - w / 2) * W
        y1 = (y - h / 2) * H
        rect = patches.Rectangle(
            (x1, y1), w * W, h * H,
            linewidth=2,
            edgecolor=PALETTE[int(cls) % len(PALETTE)],
            facecolor="none"
        )
        ax.add_patch(rect)
        class_dictionary = {0:'person', 1:'pig'}
        label = f"{class_dictionary[int(cls)]} {conf:.2f}"
        ax.text(
            x1, y1 - 4, label,
            color="white", fontsize=8, weight="bold",
            backgroundcolor=PALETTE[int(cls) % len(PALETTE)]
        )

    fig.tight_layout(pad=0)
    if save_to is not None:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_to}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("YOLOv1 inference")
    parser.add_argument("--image", type=Path, required=True, help="Путь к картинке")
    parser.add_argument("--weights", type=Path, required=True, help="checkpoint_xxx.pth")
    parser.add_argument("--out", type=Path, default="pred.png", help="Файл вывода")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    img, bboxes = predict(model, args.image, args.device, conf_thres=args.conf)
    draw_bboxes(img, bboxes, save_to=args.out)

if __name__ == "__main__":
    main()

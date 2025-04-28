import torch
import torch.nn as nn
import pandas as pd
import os
import PIL
import skimage
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter, defaultdict
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Dict, Tuple, Union, Literal
from config import *
# from model_config import architecture_config
from class_dictionary import class_dictionary
import torchvision.models as models
from model_scripts.YOLOv1 import YoloV1
from metrics.YoloLoss import YoloLoss
from metrics.meanAP import mean_average_precision
from utils.utils import Compose, get_bboxes, cellboxes_to_boxes
from utils.VOC_dataset import VOCDataset
import wandb

seed = 123
torch.manual_seed(seed)

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_mAP',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        'scheduler_factor':{
            'min': 0.01,
            'max': 0.5
        },
        'epochs': {
            'values': [10, 20, 30, 50, 100, 200]
        }
    }
}


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print(f"Save model as {filename}")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print(f"Loading model {model}")
    model.load_state_dict(checkpoint["state_dict"])

def build_parser():
    p=argparse.ArgumentParser("YOLO v1 trainer")
    p.add_argument("--train",action="store_true")
    p.add_argument("--data_dir",type=str)
    p.add_argument("--img",type=int,default=448)
    p.add_argument("--device",type=str,default="cuda")
    p.add_argument("--out",type=Path,default=Path("runs/yolov1"))
    # W&B
    p.add_argument("--wandb",action="store_true")
    p.add_argument("--project",type=str,default="yolov1")
    p.add_argument("--run-name",type=str)
    p.add_argument("--sweep", action="store_true", help="Run as a W&B sweep agent")
    return p


def plot_image_with_bboxes(img, bboxes, box_format="midpoint", labels_map=None):
    """
    img: Tensor (3, H, W)
    bboxes: List of [class, confidence, x, y, w, h] (for pred) or [class, x, y, w, h] (for GT)
    """
    img = img.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in bboxes:
        if len(box) == 6:  # prediction box
            class_label, confidence, x, y, w, h = box
            if confidence < 0.5:
                continue
        else:  # ground truth box
            class_label, x, y, w, h = box

        if box_format == "midpoint":
            x1 = x - w/2
            y1 = y - h/2
        elif box_format == "corners":
            x1, y1, x2, y2 = x, y, w, h
            w = x2 - x1
            h = y2 - y1
        else:
            raise ValueError(f"Unknown box_format {box_format}")

        rect = patches.Rectangle(
            (x1 * img.shape[1], y1 * img.shape[0]),
            w * img.shape[1],
            h * img.shape[0],
            linewidth=2,
            edgecolor="r" if len(box) == 6 else "g",
            facecolor="none",
        )
        ax.add_patch(rect)

        if labels_map is not None:
            label = labels_map.get(int(class_label), str(int(class_label)))
            ax.text(x1 * img.shape[1], y1 * img.shape[0] - 5, label, color="white", 
                    backgroundcolor="red" if len(box) == 6 else "green", fontsize=8)

    plt.axis('off')
    fig.tight_layout(pad=0)
    return fig


def train_epoch(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
    
    avg_loss = sum(mean_loss) / len(mean_loss)
    print(f"Train loss: {avg_loss}")
    return avg_loss

def train_one_run(args, config=None):
    import wandb

    with wandb.init(config=config):
        config = wandb.config

        train_dir = os.path.join(args.data_dir, 'train/')
        val_dir = os.path.join(args.data_dir, 'valid/')
        train_images = [image for image in sorted(os.listdir(train_dir))
                            if image[-4:]=='.jpg']
        
        annots = []
        for image in train_images:
            annot = image[:-4] + '.xml'
            annots.append(annot)
            
        images = pd.Series(train_images, name='images')
        annots = pd.Series(annots, name='annots')
        df = pd.concat([images, annots], axis=1)
        df = pd.DataFrame(df)

        val_images = [image for image in sorted(os.listdir(val_dir))
                            if image[-4:]=='.jpg']
        val_annots = []
        for image in val_images:
            annot = image[:-4] + '.xml'
            val_annots.append(annot)

        val_images = pd.Series(val_images, name='test_images')
        val_annots = pd.Series(val_annots, name='test_annots')
        val_df = pd.concat([val_images, val_annots], axis=1)
        val_df = pd.DataFrame(val_df)

        mobilenet = models.mobilenet_v2(weights=None)
        backbone = mobilenet.features
        model = YoloV1(backbone=backbone, split_size=7, num_boxes=2, num_classes=3, backbone_out_features=1280).to(DEVICE)

        optimizer = optim.AdamW(
            model.parameters(), lr=wandb.config.lr, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=wandb.config.scheduler_factor, patience=3, mode='max', verbose=True)
        loss_fn = YoloLoss()
        
        transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
        
        train_dataset = VOCDataset(
            df=df,
            transform=transform,
            files_dir=train_dir,
            class_dictionary=class_dictionary
        )

        test_dataset = VOCDataset(
            df=val_df,
            transform=transform, 
            files_dir=val_dir,
            class_dictionary=class_dictionary
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=wandb.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )

        val_loader = DataLoader(
            dataset=test_dataset,
            batch_size=wandb.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )

        best_val_map = 0.0

        for epoch in range(wandb.config.epochs):

            train_loss = train_epoch(train_loader, model, optimizer, loss_fn)

            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )
            train_map = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {train_map:.4f}")

            pred_boxes, target_boxes = get_bboxes(
                val_loader, model, iou_threshold=0.5, threshold=0.4
            )
            val_map = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Val mAP: {val_map:.4f}")
            
            scheduler.step(val_map)

            wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "train_mAP": train_map,
                    "val_mAP": val_map,
                    "lr": optimizer.param_groups[0]["lr"]
                })

    wandb.finish()

    


def main(argv:Iterable[str]|None=None):
    args=build_parser().parse_args(argv)

    if args.wandb and not args.sweep:
        import wandb
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "device": DEVICE,
                "scheduler_factor": SCHEDULER_FACTOR
            }
        )
    
    if args.sweep:
        import wandb
        sweep_id = wandb.sweep(sweep_config, project=args.project)

        def sweep_worker():
            train_one_run(args)

        wandb.agent(sweep_id, function=sweep_worker)
        return
    
    else:

        train_dir = os.path.join(args.data_dir, 'train/')
        val_dir = os.path.join(args.data_dir, 'valid/')
        train_images = [image for image in sorted(os.listdir(train_dir))
                            if image[-4:]=='.jpg']
        
        annots = []
        for image in train_images:
            annot = image[:-4] + '.xml'
            annots.append(annot)
            
        images = pd.Series(train_images, name='images')
        annots = pd.Series(annots, name='annots')
        df = pd.concat([images, annots], axis=1)
        df = pd.DataFrame(df)

        val_images = [image for image in sorted(os.listdir(val_dir))
                            if image[-4:]=='.jpg']
        val_annots = []
        for image in val_images:
            annot = image[:-4] + '.xml'
            val_annots.append(annot)

        val_images = pd.Series(val_images, name='test_images')
        val_annots = pd.Series(val_annots, name='test_annots')
        val_df = pd.concat([val_images, val_annots], axis=1)
        val_df = pd.DataFrame(val_df)

        mobilenet = models.mobilenet_v2(weights=None)
        backbone = mobilenet.features
        model = YoloV1(backbone=backbone, split_size=7, num_boxes=2, num_classes=3, backbone_out_features=1280).to(DEVICE)

        optimizer = optim.AdamW(
            model.parameters(), lr=wandb.config.lr, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=wandb.config.scheduler_factor, patience=3, mode='max', verbose=True)
        loss_fn = YoloLoss()
        
        transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
        
        train_dataset = VOCDataset(
            df=df,
            transform=transform,
            files_dir=train_dir,
            class_dictionary=class_dictionary
        )

        test_dataset = VOCDataset(
            df=val_df,
            transform=transform, 
            files_dir=val_dir,
            class_dictionary=class_dictionary
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=wandb.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            dataset=test_dataset,
            batch_size=wandb.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_val_map = 0.0

        for epoch in range(EPOCHS):

            train_loss = train_epoch(train_loader, model, optimizer, loss_fn)

            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )
            train_map = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {train_map:.4f}")

            pred_boxes, target_boxes = get_bboxes(
                val_loader, model, iou_threshold=0.5, threshold=0.4
            )
            val_map = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Val mAP: {val_map:.4f}")

            if args.wandb:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "train_mAP": train_map,
                    "val_mAP": val_map,
                    "lr": optimizer.param_groups[0]["lr"]
                })

                # Log sample predictions
                model.eval()
                images, labels = next(iter(val_loader))
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.no_grad():
                    preds = model(images)
                
                N = min(4, images.shape[0])

                for idx in range(N):
                    img = images[idx]
                    label = labels[idx]
                    pred = preds[idx]

                    true_bboxes = cellboxes_to_boxes(label.unsqueeze(0))[0]
                    pred_bboxes = cellboxes_to_boxes(pred.unsqueeze(0))[0]

                    fig_gt = plot_image_with_bboxes(img, true_bboxes)
                    fig_pred = plot_image_with_bboxes(img, pred_bboxes)

                    wandb.log({
                        f"Image_GT/epoch_{epoch+1}_idx_{idx}": wandb.Image(fig_gt),
                        f"Image_Pred/epoch_{epoch+1}_idx_{idx}": wandb.Image(fig_pred),
                    })
                
                model.train()
            
            scheduler.step(val_map)

            if val_map > best_val_map:
                checkpoint = {
                "state_dict": model.state_dict()
                }

                if args.wandb:
                    run_id = wandb.run.id
                    save_path = f"checkpoint_{run_id}_epoch{epoch+1}_valmap{val_map:.4f}.pth"
                else:
                    save_path = f"checkpoint_epoch{epoch+1}_valmap{val_map:.4f}.pth"

                save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
                best_val_map = val_map

    
if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from utils.intersection_over_union import intersection_over_union

class YoloLoss(nn.Module):
    """
    Calculate the loss for YOLOv1 model
    """

    def __init__(self, S=7, B=2, C=3):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S  # Grid size (S x S)
        self.B = B  # Number of bounding boxes per cell
        self.C = C  # Number of classes

        # Loss scaling factors (from YOLO v1 paper)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # Reshape to (batch_size, S, S, C + B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + 5 * self.B)

        # Extract predicted bounding boxes and class probabilities
        pred_box1 = predictions[..., self.C+1:self.C+5]  # box 1 (x, y, w, h)
        pred_box2 = predictions[..., self.C+6:self.C+10]  # box 2 (x, y, w, h)
        pred_conf1 = predictions[..., self.C]  # confidence for box 1
        pred_conf2 = predictions[..., self.C+5]  # confidence for box 2
        pred_classes = predictions[..., :self.C]  # class scores

        # Extract target bounding box and object mask
        target_box = target[..., self.C+1:self.C+5]
        object_mask = target[..., self.C].unsqueeze(3)  # 1 if object exists in cell, else 0

        # Calculate IoU for both bounding box predictions
        iou_b1 = intersection_over_union(pred_box1, target_box)
        iou_b2 = intersection_over_union(pred_box2, target_box)

        # Choose the box with higher IoU
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)

        # Select best predicted box coordinates and confidence
        best_pred_box = best_box * pred_box2 + (1 - best_box) * pred_box1
        best_pred_conf = best_box * pred_conf2.unsqueeze(-1) + (1 - best_box) * pred_conf1.unsqueeze(-1)

        #  1. Loss for Bounding Boxes

        # Apply sqrt transformation to width and height
        best_pred_box[..., 2:4] = torch.sign(best_pred_box[..., 2:4]) * torch.sqrt(torch.abs(best_pred_box[..., 2:4] + 1e-6))
        target_box[..., 2:4] = torch.sqrt(target_box[..., 2:4])

        box_loss = self.mse(
            torch.flatten(object_mask * best_pred_box, end_dim=-2),
            torch.flatten(object_mask * target_box, end_dim=-2),
        )

        #     2. Objectness loss

        object_loss = self.mse(
            torch.flatten(object_mask * best_pred_conf),
            torch.flatten(object_mask * target[..., self.C:self.C+1]),
        )

        #     3. No-Objectness loss

        no_object_loss = self.mse(
            torch.flatten((1 - object_mask) * pred_conf1.unsqueeze(-1)),
            torch.flatten((1 - object_mask) * target[..., self.C:self.C+1]),
        ) + self.mse(
            torch.flatten((1 - object_mask) * pred_conf2.unsqueeze(-1)),
            torch.flatten((1 - object_mask) * target[..., self.C:self.C+1]),
        )

        #      4. Class Loss

        class_loss = self.mse(
            torch.flatten(object_mask * pred_classes, end_dim=-2),
            torch.flatten(object_mask * target[..., :self.C], end_dim=-2),
        )

        #       5. Total Loss

        total_loss = (
            self.lambda_coord * box_loss +  # localization loss
            object_loss +                   # confidence loss for objects
            self.lambda_noobj * no_object_loss +  # confidence loss for no objects
            class_loss                      # classification loss
        )

        return total_loss
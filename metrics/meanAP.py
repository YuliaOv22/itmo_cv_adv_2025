from typing import Iterable, List, Sequence, Dict, Tuple, Union, Literal
import torch
from collections import defaultdict
from utils.intersection_over_union import intersection_over_union
Box = List[Union[int, float]]

def mean_average_precision(
    pred_boxes: List[Box],
    true_boxes: List[Box],
    iou_threshold: float = 0.5,
    box_format: Literal["midpoint", "corners"] = "midpoint",
    num_classes: int = 20,
) -> float:
    """
    Calculates Mean Average Precision (mAP) over all classes.

    Args:
        pred_boxes: List of predicted boxes in format [image_idx, class_id, score, x1, y1, x2, y2].
        true_boxes: List of ground truth boxes in same format (with score ignored).
        iou_threshold: IoU threshold to consider a detection as true positive.
        box_format: Coordinate format: "midpoint" or "corners".
        num_classes: Total number of classes.

    Returns:
        Mean Average Precision (percentage).
    """
    average_precisions = []  # store AP for each class
    epsilon = 1e-6

    for c in range(num_classes):
        # Filter boxes by class
        cls_preds = [box for box in pred_boxes if box[1] == c]
        cls_gts = [box for box in true_boxes if box[1] == c]
        total_gts = len(cls_gts)
        if total_gts == 0:
            continue

        # Track ground truth matches per image
        gts_per_image = defaultdict(list)
        for img_idx, _, *_ in cls_gts:
            gts_per_image[img_idx].append([])
        # Replace lists with match flags
        matched_flags = {
            img_idx: torch.zeros(len(boxes), dtype=torch.bool)
            for img_idx, boxes in gts_per_image.items()
        }

        # Sort predictions by confidence
        cls_preds.sort(key=lambda x: x[2], reverse=True)

        tp = torch.zeros(len(cls_preds))
        fp = torch.zeros(len(cls_preds))

        # Evaluate each prediction
        for i, pred in enumerate(cls_preds):
            img_idx, _, score, *pred_coords = pred
            pred_tensor = torch.tensor(pred_coords)
            # Get ground truths for this image
            img_gts = [gt for gt in cls_gts if gt[0] == img_idx]

            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(img_gts):
                gt_coords = torch.tensor(gt[3:])
                iou = intersection_over_union(
                    pred_tensor, gt_coords, box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            # Determine TP or FP
            if best_iou > iou_threshold:
                if not matched_flags[img_idx][best_j]:
                    tp[i] = 1  # correct detection
                    matched_flags[img_idx][best_j] = True
                else:
                    fp[i] = 1  # duplicate detection
            else:
                fp[i] = 1  # IoU too low => false positive

        # Compute precision-recall curve
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / (total_gts + epsilon)
        precisions = tp_cum / (tp_cum + fp_cum + epsilon)

        # Integrate area under PR curve
        recalls = torch.cat((torch.tensor([0.0]), recalls))
        precisions = torch.cat((torch.tensor([1.0]), precisions))
        ap = torch.trapz(precisions, recalls)
        average_precisions.append(ap)

    # Mean of APs, scaled to percentage
    if not average_precisions:
        return 0.0
    mAP = torch.mean(torch.stack(average_precisions)) * 100
    return mAP.item()
from typing import List, Union, Literal
import torch
from utils.intersection_over_union import intersection_over_union

def NMS(
    bboxes: List[List[Union[int, float]]],
    iou_threshold: float,
    score_threshold: float,
    box_format: Literal["corners", "midpoint"] = "corners"
) -> List[List[Union[int, float]]]:
    """
    Perform Non-Maximum Suppression (NMS) on a list of bounding boxes.

    Args:
        bboxes: List of boxes, each defined as [class_id, score, x1, y1, x2, y2].
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.
        score_threshold: Minimum score to keep a box.
        box_format: Format of box coordinates: "corners" or "midpoint".

    Returns:
        A list of boxes that remain after NMS.
    """
    # Filter out low-confidence boxes
    candidates = [box for box in bboxes if box[1] > score_threshold]
    # Sort boxes by descending score
    candidates.sort(key=lambda box: box[1], reverse=True)

    selected: List[List[Union[int, float]]] = []

    while candidates:
        # Pick the box with the highest score
        current = candidates.pop(0)
        selected.append(current)

        # Remove boxes of the same class with high overlap
        class_id, score, *coords = current
        current_tensor = torch.tensor(coords)

        def should_keep(box: List[Union[int, float]]) -> bool:
            other_class, other_score, *other_coords = box
            if other_class != class_id:
                return True
            iou = intersection_over_union(
                current_tensor,
                torch.tensor(other_coords),
                box_format=box_format
            )
            return iou < iou_threshold

        candidates = [box for box in candidates if should_keep(box)]

    return selected
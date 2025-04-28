import torch

def intersection_over_union(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
    box_format: str = 'midpoint',
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for batches of bounding boxes.

    Args:
        boxes_preds (torch.Tensor): Predicted boxes, shape (N, 4).
        boxes_labels (torch.Tensor): Ground truth boxes, shape (N, 4).
        box_format (str): 'midpoint' (x, y, w, h) or 'corners' (x1, y1, x2, y2).
        eps (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: IoU scores, shape (N, 1).
    """
    # Convert midpoint format to corner format if needed
    if box_format == 'midpoint':
        boxes_preds = _midpoint_to_corners(boxes_preds)
        boxes_labels = _midpoint_to_corners(boxes_labels)
    elif box_format != 'corners':
        raise ValueError(f"Unknown box_format '{box_format}'. Use 'midpoint' or 'corners'.")

    # Separate coordinates
    x1_preds, y1_preds, x2_preds, y2_preds = boxes_preds.unbind(-1)
    x1_labels, y1_labels, x2_labels, y2_labels = boxes_labels.unbind(-1)

    # Intersection boundaries
    x1_int = torch.max(x1_preds, x1_labels)
    y1_int = torch.max(y1_preds, y1_labels)
    x2_int = torch.min(x2_preds, x2_labels)
    y2_int = torch.min(y2_preds, y2_labels)

    # Intersection area
    inter_w = (x2_int - x1_int).clamp(min=0)
    inter_h = (y2_int - y1_int).clamp(min=0)
    intersection = inter_w * inter_h

    # Areas
    area_preds = (x2_preds - x1_preds).clamp(min=0) * (y2_preds - y1_preds).clamp(min=0)
    area_labels = (x2_labels - x1_labels).clamp(min=0) * (y2_labels - y1_labels).clamp(min=0)

    # Union area
    union = area_preds + area_labels - intersection + eps

    # IoU and preserve (N,1) shape
    iou = (intersection / union).unsqueeze(-1)
    return iou


def _midpoint_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2).
    """
    x_c, y_c, w, h = boxes.unbind(-1)
    half_w, half_h = w / 2, h / 2
    x1 = x_c - half_w
    y1 = y_c - half_h
    x2 = x_c + half_w
    y2 = y_c + half_h
    return torch.stack([x1, y1, x2, y2], dim=-1)
import numpy as np


def compute_distances(bboxes, points_3d, projected_uv):
    distances = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        in_bbox_mask = (
            (projected_uv[:, 0] >= x1) & (projected_uv[:, 0] <= x2) &
            (projected_uv[:, 1] >= y1) & (projected_uv[:, 1] <= y2)
        )

        if np.any(in_bbox_mask):
            points_in_bbox = points_3d[in_bbox_mask]
            dists = np.linalg.norm(points_in_bbox, axis=1)
            min_dist = float(np.min(dists))
        else:
            min_dist = None  # если нет точек внутри bbox

        distances.append(min_dist)

    return distances

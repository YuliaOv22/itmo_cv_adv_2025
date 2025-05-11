import numpy as np


def compute_distances(bboxes, points_3d, projected_points):
    distances = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        mask = (
            (projected_points[:, 0] >= x1) &
            (projected_points[:, 0] <= x2) &
            (projected_points[:, 1] >= y1) &
            (projected_points[:, 1] <= y2)
        )
        selected = points_3d[mask]
        if len(selected) > 0:
            dist = np.linalg.norm(selected[:, :3], axis=1).min()
        else:
            dist = None
        distances.append(dist)
    return distances

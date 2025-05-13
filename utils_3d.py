import json
import os
from pathlib import Path

import cv2
import numpy as np
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


def save_detections_to_json(predictions, json_path):
    results = []
    # output_data = {"results": results}

    print(predictions)
    for pred in predictions:
        print(pred)
        sample_token = pred["sample_token"]
        # results[sample_token] = []

        result = {
            'sample_token': sample_token,
            "translation": [float(x) for x in pred['translation']],
            "size": [float(x) for x in pred['size']],
            "rotation": pred['rotation'],
            "score": float(pred['detection_score']),
            "category": float(pred['detection_name'])
        }

        results.append(result)

    output_data = {"results": results}

    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def get_3d_bbox_from_2d(bbox, points_3d, threshold_distance=10.0):
    mask = (points_3d[:, 0] >= bbox[0]) & (points_3d[:, 0] <= bbox[2]) & \
           (points_3d[:, 1] >= bbox[1]) & (points_3d[:, 1] <= bbox[3])
    
    filtered_points = points_3d[mask]

    if len(filtered_points) == 0:
        return None

    min_x, min_y, min_z = np.min(filtered_points, axis=0)
    max_x, max_y, max_z = np.max(filtered_points, axis=0)

    return [min_x, min_y, min_z, max_x - min_x, max_y - min_y, max_z - min_z]


def get_lidar_points_in_bbox(bbox, projected_uv, points_3d):
    x1, y1, x2, y2 = bbox
    mask = (projected_uv[:, 0] >= x1) & (projected_uv[:, 0] <= x2) & \
           (projected_uv[:, 1] >= y1) & (projected_uv[:, 1] <= y2)
    return points_3d[mask]


def estimate_3d_bbox_from_points(points):
    if points.shape[0] < 5:
        return None

    center = np.mean(points, axis=0)

    pca = PCA(n_components=3)
    pca.fit(points - center)
    axes = pca.components_

    proj = np.dot(points - center, axes.T)
    size = np.max(proj, axis=0) - np.min(proj, axis=0)

    rotation_matrix = axes.T
    quat = R.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]

    bbox = {
        'translation': center.tolist(),
        'size': size.tolist(),
        'rotation': quat_wxyz,
        'name': 'car'
    }
    return bbox


def evaluate_3d_detections(nusc, pred_json_path, scene, console, args):
    
    output_dir = Path(f"nusc_eval_scene_{args.scene:02d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_config = config_factory('detection_cvpr_2019')

    split_name = f"scene_{args.scene:02d}_only"
    split_scenes = {split_name: [scene['name']]}

    console.print(f"🔍 Запускаем оценку сцены: {scene['name']}")

    detection_eval = DetectionEval(
        nusc=nusc,
        config=eval_config,
        result_path=pred_json_path,
        eval_set='mini_val',
        output_dir=str(output_dir),
        verbose=True,
    )

    detection_eval.main()
    metrics_path = output_dir / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        console.print(f"\n[bold cyan]Результаты оценки для сцены {scene['name']}:[/bold cyan]\n")
        for key, value in metrics.items():
            console.print(f"{key:<25} : {value:.4f}")
    else:
        console.print("[red]Не удалось найти файл с метриками![/red]")

    console.print(f"\n[green]Оценка завершена.[/green] Результаты сохранены в: {output_dir}")

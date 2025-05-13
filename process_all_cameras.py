import argparse
import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
from nuscenes.nuscenes import NuScenes
from rich.console import Console
from tqdm import tqdm

from distances_utils import *
from nuscenes_utils import *
from projection_utils import *
from utils_3d import *
from visualize import *

yolo_lock = Lock()

import numpy as np
from ultralytics import YOLO

console = Console()

CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]


def load_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=0, help='Номер сцены')
    parser.add_argument('--threshold_near', type=float, default=10.0, help='Порог для ближних точек (красный)')
    parser.add_argument('--threshold_mid', type=float, default=25.0, help='Порог для средних точек (оранжевый)')
    return parser.parse_args()


def run_yolo(yolo_model, image):
    with yolo_lock:
        results = yolo_model(image, verbose=False)[0]
    bboxes = []
    classes = []
    confs = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bboxes.append([x1, y1, x2, y2])
        classes.append(int(box.cls[0]))
        confs.append(float(box.conf[0]))
    return np.array(bboxes), np.array(classes), np.array(confs)


def process_camera_view(nusc, cam_token, lidar_token, yolo_model, args):
    image, cam_data = load_image(nusc, cam_token)
    lidar_pc, lidar_data = load_lidar(nusc, lidar_token)

    bboxes, classes, confidences = run_yolo(yolo_model, image)
    cam_intrinsic, lidar2cam = get_calibration(nusc, cam_data, lidar_data)
    projected_uv, points_3d = project_lidar2image(lidar_pc, lidar2cam, cam_intrinsic)

    mask = create_bbox_mask(image.shape[:2], bboxes)
    points_3d_filtered = filter_points_by_mask(projected_uv, points_3d, mask)
    mask_3d = np.isin(points_3d, points_3d_filtered).all(axis=1)
    projected_uv_filtered = projected_uv[mask_3d]
    distances = compute_distances(bboxes, points_3d_filtered, projected_uv_filtered)

    img_with_points = draw_projected_points(
        image.copy(), projected_uv_filtered, points_3d_filtered,
        threshold_near=args.threshold_near, threshold_mid=args.threshold_mid
    )
    img_final_2d = draw_detections(
        img_with_points.copy(), bboxes, classes, confidences, distances
    )

    img_final_3d = image.copy()
    predictions_3d = []

    for bbox, cls, score in zip(bboxes, classes, confidences):
        inside_points = get_lidar_points_in_bbox(bbox, projected_uv_filtered, points_3d_filtered)
        if len(inside_points) < 5:
            continue

        bbox_3d = estimate_3d_bbox_from_points(inside_points)
        if bbox_3d is not None:
            prediction_dict = {
                "sample_token": lidar_data['sample_token'],
                "translation": bbox_3d["translation"],
                "size": bbox_3d["size"],
                "rotation": bbox_3d["rotation"],
                "detection_name": cls,
                "detection_score": float(score),
                "attribute_name": ""
            }
            predictions_3d.append(prediction_dict)

    return img_final_2d, img_final_3d, predictions_3d


def main(args, data_path='~/nuscenes', version='v1.0-mini'):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    scenes = nusc.scene

    if args.scene < 0 or args.scene >= len(scenes):
        console.print(f"Сцена с номером {args.scene} не найдена. Доступно: 0 - {len(scenes) - 1}")
        return

    scene = scenes[args.scene]
    sample_token = scene['first_sample_token']

    yolo_model = load_yolo('yolov8l.pt')
    frame_id = 0
    all_predictions = []

    with tempfile.TemporaryDirectory() as tmpdir:
        console.print(f"Сохраняем кадры в: {tmpdir}")


        with tqdm(desc="Обрабатываем кадры", unit="frame") as pbar:
            i = 0
            while sample_token:
                sample = nusc.get('sample', sample_token)
                lidar_token = sample['data']['LIDAR_TOP']

                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = {
                        cam: executor.submit(
                            process_camera_view,
                            nusc,
                            sample['data'][cam],
                            lidar_token,
                            yolo_model,
                            args
                        ) for cam in CAMERAS
                    }

                    images = {}
                    images_3d = {}
                    for cam, future in futures.items():
                        try:
                            img_final, img_3d, preds_3d = future.result()
                            images[cam] = img_final
                            images_3d[cam] = img_3d
                            all_predictions.extend(preds_3d)
                        except Exception as e:
                            console.print(f"Ошибка обработки {cam}: {e}")
                            traceback.print_exc()
                            images[cam] = np.zeros((900, 1600, 3), dtype=np.uint8)
                            images_3d[cam] = np.zeros((900, 1600, 3), dtype=np.uint8)
                    if i == 2:
                        break
                    i+= 1

                img_grid = build_camera_grid(images, cameras=CAMERAS)
                frame_path = os.path.join(tmpdir, f"frame_{frame_id:04d}_grid.jpg")
                cv2.imwrite(frame_path, img_grid)
                pbar.update(1)

                sample_token = sample['next']
                frame_id += 1

        output_video = f'./videos/scene_{args.scene:02d}_demo_video.gif'
        save_frames2gif(tmpdir, output_path=output_video)
        console.print(f"Видео сохранено: {output_video}")

    # pred_json_path = f"./outputs/scene_{args.scene:02d}_detections.json"
    # save_detections_to_json(all_predictions, pred_json_path)
    # console.print(f"Предсказания 3D боксов сохранены: {pred_json_path}")
    # evaluate_3d_detections(nusc, pred_json_path, scene, console, args)


if __name__ == "__main__":
    args = parse_args()
    main(args, data_path='./data/')

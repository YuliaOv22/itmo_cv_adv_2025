import os
import cv2
import tempfile
import argparse

from rich.console import Console

from distances_utils import *
from nuscenes_utils import *
from projection_utils import *
from visualize import *
from yolo_utils import *

console = Console()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=0, help='Номер сцены')
    parser.add_argument('--camera', type=str, default='CAM_FRONT', help='Камера')
    parser.add_argument('--threshold_near', type=float, default=10.0, help='Порог для ближних точек (красный)')
    parser.add_argument('--threshold_mid', type=float, default=25.0, help='Порог для средних точек (оранжевый)')
    return parser.parse_args()


def main(args, data_path='~/nuscenes', version='v1.0-mini'):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    scenes = nusc.scene
    if args.scene < 0 or args.scene >= len(scenes):
        console.print(f"[red]Сцена с номером {args.scene} не найдена. Доступно: 0 - {len(scenes) - 1}[/red]")
        return

    scene = scenes[args.scene]
    scene_token = scene['token']
    sample_token = scene['first_sample_token']

    yolo_model = load_yolo('yolov8l.pt')
    frame_id = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        console.print(f"[green]Сохраняем кадры в: {tmpdir}[/green]")

        while sample_token:
            sample = nusc.get('sample', sample_token)

            image, cam_data = load_image(nusc, sample['data'][args.camera])
            lidar_pc, lidar_data = load_lidar(nusc, sample['data']['LIDAR_TOP'])

            bboxes, classes, confidences = run_yolo(yolo_model, image)
            cam_intrinsic, lidar2cam = get_calibration(nusc, cam_data, lidar_data)
            projected_uv, points_3d = project_lidar2image(lidar_pc, lidar2cam, cam_intrinsic)

            mask = create_bbox_mask(image.shape[:2], bboxes)
            points_3d_filtered = filter_points_by_mask(projected_uv, points_3d, mask)

            mask_3d = np.isin(points_3d, points_3d_filtered).all(axis=1)
            projected_uv_filtered = projected_uv[mask_3d]

            distances = compute_distances(bboxes, points_3d_filtered, projected_uv_filtered)

            img_with_points = draw_projected_points(
                image.copy(),
                projected_uv_filtered, 
                points_3d_filtered,
                threshold_near=args.threshold_near,
                threshold_mid=args.threshold_mid
            )

            frame_path1 = os.path.join(tmpdir, f"frame_{frame_id:04d}_points.jpg")
            frame_path2 = os.path.join(tmpdir, f"frame_{frame_id:04d}_detections.jpg")

            cv2.imwrite(frame_path1, img_with_points)

            img_final = draw_detections(
                img_with_points.copy(), bboxes, classes, confidences, distances
            )
            cv2.imwrite(frame_path2, img_final)

            console.print(f"[blue]Кадр {frame_id} сохранён.[/blue]")

            sample_token = sample['next']
            frame_id += 1

        output_video = f'scene_{args.scene:02d}_{args.camera}_demo_video.mp4'
        save_frames2video(tmpdir, output_path=output_video)
        console.print(f"[bold green]Видео сохранено: {output_video}[/bold green]")


if __name__ == "__main__":
    args = parse_args()
    main(args, data_path='./data/')

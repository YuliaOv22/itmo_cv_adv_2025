import os
import shutil
from glob import glob

import cv2
import numpy as np


def draw_detections(image, bboxes, classes, confidences, distances=None):
    for bbox, cls, conf, dist in zip(bboxes, classes, confidences, distances):
        x1, y1, x2, y2 = map(int, bbox)
        label = f'{cls} {conf:.2f}'
        if (distances is not None) and (dist is not None):
            label += f' | {dist:.1f} m'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image

# def draw_projected_points(image, projected_uv, color=(0, 0, 255)):
#     for u, v in projected_uv.astype(int):
#         if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
#             cv2.circle(image, (u, v), radius=1, color=color, thickness=-1)
#     return image

def draw_projected_points(image, projected_uv, points_3d, threshold_near=10, threshold_mid=25):
    """
    Отрисовывает проецированные точки на изображении.
    Цвет зависит от расстояния до каждой точки:
        🔴 ближе threshold_near
        🟠 между threshold_near и threshold_mid
        🟢 дальше threshold_mid
    """
    for (u, v), pt in zip(projected_uv.astype(int), points_3d):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            # print(pt, np.linalg.norm(pt))
            distance = np.linalg.norm(pt)
            if distance < threshold_near:
                color = (0, 0, 255)       # Красный
            elif distance < threshold_mid:
                color = (0, 165, 255)     # Оранжевый
            else:
                color = (0, 255, 0)       # Зелёный
            cv2.circle(image, (u, v), radius=2, color=color, thickness=-1)
    return image


# def draw_projected_points(image, projected_uv, distances, threshold_near=10, threshold_mid=25):
#     """
#     Рисует проецированные точки цветом, зависящим от расстояния до ближайшего объекта.
#     Пропускает точки с неопределённым расстоянием (None).
#     """
#     for (u, v), dist in zip(projected_uv.astype(int), distances):
#         if dist is None:
#             continue  # пропускаем, если нет расстояния
#         if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
#             if dist < threshold_near:
#                 color = (0, 0, 255)       # 🔴 Красный
#             elif dist < threshold_mid:
#                 color = (0, 165, 255)     # 🟠 Оранжевый
#             else:
#                 color = (0, 255, 0)       # 🟢 Зелёный
#             cv2.circle(image, (u, v), radius=2, color=color, thickness=-1)
#     return image


def stack_camera_views(images_dict, image_size=(640, 360)):
    """
    Склеивает изображения камер в сетку 2x3 (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
    """
    names = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'
    ]
    
    images = [cv2.resize(images_dict.get(name, np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)), image_size)
              for name in names]
    
    row1 = np.hstack(images[:3])
    row2 = np.hstack(images[3:])
    grid = np.vstack([row1, row2])
    
    return grid


# def save_frames2video(frame_dir, output_path='demo_video.mp4', fps=5):
#     frame_paths = sorted(glob(os.path.join(frame_dir, '*.jpg')))
#     if not frame_paths:
#         print("❌ No frames to compile.")
#         return
#     first_frame = cv2.imread(frame_paths[0])
#     height, width = first_frame.shape[:2]

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     for path in frame_paths:
#         frame = cv2.imread(path)
#         out.write(frame)

#     out.release()


def save_frames2video(frames_dir, output_path, fps=10):
    import glob
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*_grid.jpg')))

    if not frame_files:
        print("⚠️ Нет кадров для видео.")
        return

    sample_frame = cv2.imread(frame_files[0])
    height, width = sample_frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        writer.write(frame)

    writer.release()
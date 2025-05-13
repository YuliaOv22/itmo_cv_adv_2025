import os
from glob import glob

import cv2
import numpy as np
import imageio


def draw_detections(image, bboxes, classes, confidences, distances=None):
    for bbox, cls, conf, dist in zip(bboxes, classes, confidences, distances):
        x1, y1, x2, y2 = map(int, bbox)
        label = f'{cls} {conf:.2f}'
        if (distances is not None) and (dist is not None):
            label += f' | {dist:.1f} m'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image


def draw_projected_points(image, projected_uv, points_3d, threshold_near=10, threshold_mid=25):
    for (u, v), pt in zip(projected_uv.astype(int), points_3d):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            # print(pt, np.linalg.norm(pt))
            distance = np.linalg.norm(pt)
            if distance < threshold_near:
                color = (0, 0, 255)
            elif distance < threshold_mid:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0) 
            cv2.circle(image, (u, v), radius=2, color=color, thickness=-1)
    return image


def stack_camera_views(images_dict, image_size=(640, 360)):
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


def save_frames2video(frames_dir, output_path, fps=10):
    import glob
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*_grid.jpg')))

    if not frame_files:
        print("Нет кадров для видео.")
        return

    sample_frame = cv2.imread(frame_files[0])
    height, width = sample_frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        writer.write(frame)

    writer.release()


def save_frames2gif(frames_dir, output_path, fps=10):
    frame_files = sorted(glob(os.path.join(frames_dir, '*_grid.jpg')))
    
    if not frame_files:
        print("Нет кадров для GIF.")
        return

    frames = []
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def build_camera_grid(images, cameras):
    def resize(img):
        return cv2.resize(img, (640, 360)) if img.shape[1] != 640 else img

    row1 = np.hstack([resize(images.get(cam, np.zeros((360, 640, 3), np.uint8))) for cam in cameras[:3]])
    row2 = np.hstack([resize(images.get(cam, np.zeros((360, 640, 3), np.uint8))) for cam in cameras[3:]])
    return np.vstack([row1, row2])

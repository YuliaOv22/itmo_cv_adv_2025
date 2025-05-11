import cv2
from networkx import dag_longest_path

from distances_utils import *
from nuscenes_utils import *
from projection_utils import *
from visualize import *
from yolo_utils import *


def main():
    nusc = load_nuscenes(data_path='./data/')
    sample = get_sample(nusc)
    
    image, cam_data = load_image(nusc, sample['data']['CAM_FRONT'])
    cv2.imwrite('./figs/original_image.png', image)
    lidar_pc, lidar_data = load_lidar(nusc, sample['data']['LIDAR_TOP'])

    yolo_model = load_yolo('yolov8l.pt')
    bboxes, classes, confidences = run_yolo(yolo_model, image)

    image = draw_detections(image, bboxes, classes, confidences, distances=[None]*len(confidences))
    cv2.imwrite('./figs/img_w_detections.png', image)


    cam_intrinsic, lidar_to_cam = get_calibration(nusc, cam_data, lidar_data)
    projected_uv, points_3d = project_lidar_to_image(lidar_pc, lidar_to_cam, cam_intrinsic)
    distances = compute_distances(bboxes, points_3d, projected_uv)
    image = draw_detections(image, bboxes, classes, confidences, distances)
    cv2.imwrite('./figs/img_w_distances.png', image)


    # cv2.imshow('Detection', image)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
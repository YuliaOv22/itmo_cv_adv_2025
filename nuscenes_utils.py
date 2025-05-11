import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def load_nuscenes(data_path='~/nuscenes', version='v1.0-mini'):
    return NuScenes(version=version, dataroot=data_path, verbose=True)


def get_sample(nusc, scene_index=0, sample_index=10):
    scene = nusc.scene[scene_index]
    first_sample = nusc.get('sample', scene['first_sample_token'])
    sample = first_sample
    for _ in range(sample_index):
        sample = nusc.get('sample', sample['next'])
    return sample


def load_image(nusc, cam_data_token):
    cam_data = nusc.get('sample_data', cam_data_token)
    img = cv2.imread(nusc.get_sample_data_path(cam_data_token))
    return img, cam_data


def load_lidar(nusc, lidar_token):
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(lidar_path)
    return pc, lidar_data

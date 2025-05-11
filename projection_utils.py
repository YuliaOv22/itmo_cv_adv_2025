import numpy as np
from pyquaternion import Quaternion


def create_bbox_mask(image_shape, bboxes):
    """
    Создаёт бинарную маску для всех bounding box'ов.

    Параметры:
    - image_shape: форма изображения (H, W)
    - bboxes: список bounding box'ов [x1, y1, x2, y2]

    Возвращает:
    - mask: бинарная маска изображения, где 1 — внутри bbox, 0 — вне
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        mask[y1:y2+1, x1:x2+1] = 1
    return mask


def filter_points_by_mask(uv, points_3d, mask):
    """
    Фильтрует 3D-точки, оставляя только те, чьи 2D-координаты попадают в маску.

    Параметры:
    - uv: 2D-координаты точек (N, 2)
    - points_3d: 3D-координаты точек (N, 3)
    - mask: бинарная маска изображения

    Возвращает:
    - filtered_points_3d: отфильтрованные 3D-точки
    """
    H, W = mask.shape
    uv_int = np.round(uv).astype(int)

    valid = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
        (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    uv_int = uv_int[valid]
    points_3d = points_3d[valid]

    mask_values = mask[uv_int[:, 1], uv_int[:, 0]]
    return points_3d[mask_values == 1]


def get_calibration(nusc, cam_data, lidar_data):
    """
    Получает калибровочные параметры камеры и лидара.

    Параметры:
    - nusc: объект NuScenes
    - cam_data: метаданные камеры
    - lidar_data: метаданные лидара

    Возвращает:
    - cam_intrinsic: матрица интринсиков камеры 3x3
    - lidar2cam: матрица преобразования 4x4 из лидара в координаты камеры
    """
    cam_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    cam_intrinsic = np.array(cam_sensor['camera_intrinsic'])

    lidar2ego_rot = Quaternion(lidar_sensor['rotation']).rotation_matrix
    lidar2ego_trans = np.array(lidar_sensor['translation'])

    cam2ego_rot = Quaternion(cam_sensor['rotation']).rotation_matrix
    cam2ego_trans = np.array(cam_sensor['translation'])

    R = cam2ego_rot @ lidar2ego_rot.T
    t = cam2ego_trans - R @ lidar2ego_trans

    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = R
    lidar2cam[:3, 3] = t

    return cam_intrinsic, lidar2cam


def project_lidar2image(pc, lidar2cam, cam_intrinsic):
    """
    Проецирует 3D-точки из облака лидара на 2D-изображение.

    Параметры:
    - pc: облако точек лидара (объект LidarPointCloud)
    - lidar2cam: матрица 4x4 преобразования из лидара в камеру
    - cam_intrinsic: матрица интринсиков камеры 3x3

    Возвращает:
    - uv: 2D-координаты точек на изображении (N, 2)
    - pts_cam: 3D-координаты точек в системе камеры (N, 3)
    """
    points_lidar = pc.points[:3, :]
    points_lidar_h = np.vstack((points_lidar, np.ones(points_lidar.shape[1])))

    points_cam = lidar2cam @ points_lidar_h
    valid = points_cam[2, :] > 0.1
    points_cam = points_cam[:, valid]

    uv_hom = cam_intrinsic @ points_cam[:3, :]
    uv = uv_hom[:2, :] / uv_hom[2, :]

    return uv.T, points_cam[:3, :].T
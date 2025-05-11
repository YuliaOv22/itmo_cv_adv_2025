import numpy as np
from pyquaternion import Quaternion


def get_calibration(nusc, cam_data, lidar_data):
    """
    Получает калибровочные параметры: интринсики камеры и матрицу преобразования из координат лидара в координаты камеры.
    
    Parameters:
    - nusc: объект NuScenes
    - cam_data: словарь с метаданными изображения камеры
    - lidar_data: словарь с метаданными облака точек лидара

    Returns:
    - cam_intrinsic: матрица 3x3 с внутренними параметрами камеры
    - lidar_to_cam: матрица 4x4 преобразования из лидара в координаты камеры
    """
    
    # Получаем калибровочные параметры камеры и лидара
    cam_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    # Внутренние параметры камеры (intrinsic matrix)
    cam_intrinsic = np.array(cam_sensor['camera_intrinsic'])  # shape: (3, 3)

    # Поворот и сдвиг: из лидара в координаты ego-авто
    lidar_to_ego_rot = Quaternion(lidar_sensor['rotation']).rotation_matrix  # shape: (3, 3)
    lidar_to_ego_trans = np.array(lidar_sensor['translation'])  # shape: (3,)

    # Поворот и сдвиг: из камеры в координаты ego-авто
    cam_to_ego_rot = Quaternion(cam_sensor['rotation']).rotation_matrix
    cam_to_ego_trans = np.array(cam_sensor['translation'])

    # Теперь хотим построить преобразование: lidar -> ego -> cam
    # Ищем lidar_to_cam = cam_to_ego⁻¹ * lidar_to_ego

    # То есть: R_cam_ego * R_ego_lidar = R_cam_lidar (так как ортогональные матрицы, инверсия == транспонирование)
    R = cam_to_ego_rot @ lidar_to_ego_rot.T

    # t = t_cam - R * t_lidar (сдвиг между сенсорами в системе камеры)
    t = cam_to_ego_trans - R @ lidar_to_ego_trans

    # Формируем итоговую матрицу преобразования (гомогенная матрица 4x4)
    lidar_to_cam = np.eye(4)
    lidar_to_cam[:3, :3] = R
    lidar_to_cam[:3, 3] = t

    return cam_intrinsic, lidar_to_cam


def project_lidar_to_image(pc, lidar_to_cam, cam_intrinsic):
    """
    Проецирует 3D-точки из лидара на 2D-изображение с помощью интринсиков и матрицы преобразования.

    Parameters:
    - pc: облако точек (obj from nuscenes.utils.data_classes.LidarPointCloud)
    - lidar_to_cam: матрица 4x4 для преобразования lidar → camera
    - cam_intrinsic: 3x3 камера-интринсики

    Returns:
    - uv: координаты точек на изображении (Nx2)
    - pts_cam: координаты точек в системе камеры (Nx3)
    """

    # 1. Берём только XYZ координаты (игнорируем отражение/интенсивность)
    points_lidar = pc.points[:3, :]  # shape: (3, N)

    # 2. Добавляем строчку из единиц для перехода к гомогенным координатам
    points_lidar_h = np.vstack((points_lidar, np.ones(points_lidar.shape[1])))  # shape: (4, N)

    # 3. Преобразуем точки в координаты камеры: lidar → cam
    points_cam = lidar_to_cam @ points_lidar_h  # shape: (4, N)

    # 4. Оставляем только те точки, что находятся перед камерой (Z > 0)
    valid = points_cam[2, :] > 0
    points_cam = points_cam[:, valid]

    # 5. Применяем матрицу интринсиков для получения 2D проекции
    uv_hom = cam_intrinsic @ points_cam[:3, :]  # shape: (3, N)

    # 6. Делим на Z (перспективное деление) → получаем координаты на изображении
    uv = uv_hom[:2, :] / uv_hom[2, :]

    # 7. Возвращаем: (x, y) точки на изображении + (X, Y, Z) координаты в камере
    return uv.T, points_cam[:3, :].T  # (N, 2), (N, 3)

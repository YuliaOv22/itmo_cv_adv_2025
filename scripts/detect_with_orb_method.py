import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_images(
    images_paths: Path, target_size: Tuple[int, int] = (128, 128)
) -> List[np.ndarray]:
    """Загружает изображения из указанного пути и изменяет их размер."""

    images = []

    for path in tqdm(images_paths, desc="Загрузка изображений"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            images.append(img_resized)

    return np.array(images)


def orb_compare(image1: np.ndarray, image2: np.ndarray, threshold: int = 10) -> bool:
    """Сравнивает два изображения с помощью ORB-дескрипторов."""

    # Инициализируем ORB детектор
    orb = cv2.ORB_create()

    # BFMatcher с Hamming расстоянием (так как ORB выдаёт бинарные дескрипторы)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Вычисление ключевых точек и дескрипторов
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        print(f"Не найдены дескрипторы для одного из изображений.")
        return False

    # Поиск совпадений
    matches = bf.match(des1, des2)

    # Фильтрация хороших совпадений
    good_matches = [m for m in matches if m.distance < 60]

    return len(good_matches) >= threshold


def get_image_comparison(
    train_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_paths: List[Path],
    threshold: int = 10,
) -> List[Path]:
    """Сравнивает изображения из папок train и test с использованием метода ORB."""

    unwanted_images = []

    for item, test_img in enumerate(
        tqdm(test_images, desc="Обработка тестовых изображений")
    ):

        found = False
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    orb_compare, train_img, test_img, threshold
                ): train_img  # Сохраняем train_img как ключ
                for train_img in train_images  # Создаем задачи для всех train_img
            }

            for future in as_completed(
                futures
            ):  # Обрабатываем задачи по мере завершения
                if future.result():
                    unwanted_images.append(test_paths[item])
                    found = True
                    break  # Выходим при первом совпадении

            # Отменяем оставшиеся задачи
            for future in futures:
                future.cancel()

        if found:
            continue

    print(
        f"Количество лишних изображений согласно предсказаниям: {len(unwanted_images)}"
    )

    return unwanted_images

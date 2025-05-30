import cv2
from pathlib import Path
import numpy as np
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time



def load_sift_descriptors(image_paths: list[Path]) -> list[Optional[np.ndarray]]:
    """Предварительно загружает SIFT-дескрипторы для всех изображений."""
    descriptors = []
    for path in tqdm(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
    return descriptors

def compare_descriptors_sift(
    train_descriptor: np.ndarray,
    test_descriptor: np.ndarray,
    threshold: int
) -> bool:
    """Сравнивает два SIFT-дескриптора и определяет, являются ли изображения похожими. """
    if train_descriptor is None or test_descriptor is None:
        return False

    # Используем BFMatcher с L2-нормой (подходит для SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(train_descriptor, test_descriptor)
    good_matches = [m for m in matches if m.distance < 0.75 * max(1, np.median([m.distance for m in matches]))]
    return len(good_matches) >= threshold



def get_image_comparison(
    train_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_paths: List[Path],
    threshold: int = 10,
) -> List[Path]:
    """Сравнивает изображения из папок train и test с использованием метода ORB."""

    start_time = time.time()
    unwanted_images = []

    for item, test_img in enumerate(
        tqdm(test_images, desc="Обработка тестовых изображений")
    ):

        found = False
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    compare_descriptors_sift, train_img, test_img, threshold
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
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 1000:
            print("Время выполнения превысило 1000 секунд. Прерывание выполнения.")
            break

    print(
        f"Количество лишних изображений согласно предсказаниям: {len(unwanted_images)}"
    )


    return unwanted_images
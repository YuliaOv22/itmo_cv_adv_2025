import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_images(
    images_paths: Path, target_size: Tuple[int, int] = (128, 128)
) -> List[np.ndarray]:
    """Загружает изображения из указанного пути и изменяет их размер."""

    images = []

    for path in images_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            images.append(img_resized)

    return np.array(images)


def get_ssim(image1: np.ndarray, image2: np.ndarray, threshold: int) -> bool:
    """Возвращает True, если изображения похожи, иначе False по методу SSIM."""
    similarity = ssim(image1, image2)
    return similarity >= threshold


def get_image_comparison(
    train_images: List[np.ndarray],
    test_images: List[np.ndarray],
    test_paths: List[Path],
    threshold: float = 0.95,
) -> List[Path]:
    """Сравнивает изображения из папок train и test с использованием метода SSIM."""

    unwanted_images = []

    for item, test_img in enumerate(
        tqdm(test_images, desc="Обработка тестовых изображений")
    ):

        found = False
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    get_ssim, train_img, test_img, threshold
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

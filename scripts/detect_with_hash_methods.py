from PIL import Image
from pathlib import Path
from typing import List
import imagehash
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache


@lru_cache(maxsize=1024)
def _get_phash(image_path: Path) -> int:
    """Возвращает pHash изображения из кэша или вычисляет его."""
    return imagehash.phash(Image.open(image_path))


@lru_cache(maxsize=1024)
def _get_dhash(image_path: Path) -> int:
    """Возвращает dHash изображения из кэша или вычисляет его."""
    return imagehash.dhash(Image.open(image_path))


@lru_cache(maxsize=1024)
def _get_fast_dhash(image_path: Path) -> int:
    """Возвращает dHash изображения из кэша или вычисляет его, но с меньшим размером хэша."""
    img_processed = Image.open(image_path).convert("L").resize((9, 8), Image.BILINEAR)
    return imagehash.dhash(img_processed, hash_size=8)


def compare_hashes(
    imege_path1: Path, imege_path2: Path, threshold: int, comparison_method: str
) -> bool:
    """Сравнивает две картинки по их хэшам."""

    if comparison_method == "phash":
        hash1 = _get_phash(imege_path1)
        hash2 = _get_phash(imege_path2)
        return hash1 - hash2 < threshold

    if comparison_method == "dhash":
        hash1 = _get_dhash(imege_path1)
        hash2 = _get_dhash(imege_path2)
        return hash1 - hash2 < threshold

    if comparison_method == "fast_dhash":
        hash1 = _get_fast_dhash(imege_path1)
        hash2 = _get_fast_dhash(imege_path2)
        return hash1 - hash2 < threshold


def get_image_comparison(
    train_images: List[Path],
    test_images: List[Path],
    threshold: int,
    comparison_method: str,
) -> List[Path]:
    """Сравнивает изображения из папок train и test с использованием указанного метода сравнения."""

    unwanted_images = []

    for test_img in tqdm(test_images, desc="Обработка тестовых изображений"):

        found = False
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    compare_hashes, train_img, test_img, threshold, comparison_method
                ): train_img  # Сохраняем train_img как ключ
                for train_img in train_images  # Создаем задачи для всех train_img
            }

            for future in as_completed(
                futures
            ):  # Обрабатываем задачи по мере завершения
                if future.result():
                    unwanted_images.append(test_img)
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

import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm


def load_histograms(image_paths: list[Path], hist_size: int) -> np.ndarray:
    """Возвращает цветовые гистограммы для изображений."""

    histograms = []

    for path in image_paths:
        hist = _compute_color_histogram(path, hist_size=hist_size)
        if hist is not None:
            histograms.append(hist)

    return np.array(histograms)


def _compute_color_histogram(image_path: Path, hist_size: int = 32) -> np.ndarray:
    """Вычисляет цветовую гистограмму для изображения."""

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Преобразование в HSV
    hist = cv2.calcHist(
        [img_hsv], [0, 1], None, [hist_size, hist_size], [0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)  # Нормализация гистограммы

    return hist.flatten()


def get_histogram_comparison(
    train_hists: list[np.ndarray],
    test_hists: list[np.ndarray],
    test_paths: list[Path],    
    threshold: float = 0.85,
    method: str = "correl",
) -> list[Path]:
    """Сравнивает изображения из папок train и test с использованием гистограмм."""

    start_time = time.time()
    unwanted_images = []

    for item, test_hist in enumerate(
        tqdm(test_hists, desc="Обработка тестовых изображений")
    ):
        found_match = False
        for train_hist in train_hists:
            if method == "correl":
                similarity = cv2.compareHist(
                    test_hist, train_hist, method=cv2.HISTCMP_CORREL
                )
                if similarity >= threshold:
                    found_match = True
                    break
            if method == "intersect":
                similarity = cv2.compareHist(
                    test_hist, train_hist, method=cv2.HISTCMP_INTERSECT
                )
                if similarity >= threshold:
                    found_match = True
                    break
            if method == "chisqr":
                similarity = cv2.compareHist(
                    test_hist, train_hist, method=cv2.HISTCMP_CHISQR
                )
                if similarity <= threshold:
                    found_match = True
                    break

        if found_match:
            unwanted_images.append(test_paths[item])

    print(
        f"Количество лишних изображений согласно предсказаниям: {len(unwanted_images)}"
    )

    end_time = time.time()
    final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
    print(f"Время выполнения: {final_time}")

    return unwanted_images

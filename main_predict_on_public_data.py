from pathlib import Path
from metrics.compare_images_metrics import compute_metrics
from utils.read_save_files import read_images_from_directory, save_paths_to_file
import scripts.detect_with_hash_methods as hm
import scripts.detect_with_ssim_method as ssm
import scripts.detect_with_histogram_method as hgm
import scripts.detect_with_orb_method as om
import scripts.detect_with_sift_method as sm
from scripts.detect_with_cnn import TransformerEmbedder
from utils.log import TeeLoggerContext
from utils.measure_time import measure_time
import time


def open_files(
    train_images_path, test_images_path, image_extensions, leakage_images_path
):
    """Функция для чтения изображений из директорий и лишних изображений из файла."""

    # Чтение изображений из папок
    train_images, test_images = read_images_from_directory(
        train_images_path, test_images_path, image_extensions
    )

    # Чтение лишних изображений из файла
    with open(leakage_images_path, "r", encoding="utf-8") as file:
        leakage_images = file.read().split("\n")
    print(
        f"Количество лишних изображений в оригинальной разметке: {len(leakage_images)}"
    )

    return train_images, test_images, leakage_images


if __name__ == "__main__":

    log_path = Path("logs/predict_on_public_data.log")
    log_path.parent.mkdir(exist_ok=True)

    with TeeLoggerContext(log_path):

        print("-------------------Public data-------------------")
        # Настройка переменных
        input_dir_path = Path("public_data")
        train_images_path = f"{input_dir_path}/train"
        test_images_path = f"{input_dir_path}/test"
        leakage_images_path = f"{input_dir_path}/leakage_files.txt"
        output_dir_path = "output"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # Метод pHash (Perceptual Hash)
        print("-------------------1. Метод pHash (Perceptual Hash)-------------------")
        start_time = time.time()
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        output_path = Path(f"{output_dir_path}/pred_cached_pHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="phash"
        )        
        end_time = time.time()
        phash_preds_time = measure_time(start_time, end_time)
        metrics_phash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # phash_preds = preds

        # Метод dHash (Difference Hash)
        print("-------------------2. Метод dHash (Difference Hash)-------------------")
        start_time = time.time()
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        output_path = Path(f"{output_dir_path}/pred_cached_dHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="dhash"
        )
        end_time = time.time()
        dhash_preds_time = measure_time(start_time, end_time)
        metrics_dhash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # dhash_preds = preds

        # Метод Fast dHash
        print("-------------------3. Метод Fast dHash-------------------")
        start_time = time.time()
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        output_path = Path(f"{output_dir_path}/pred_cached_fast_dHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="fast_dhash"
        )
        end_time = time.time()
        fast_dhash_preds_time = measure_time(start_time, end_time)
        metrics_fast_dhash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        fast_dhash_preds = preds

        # Метод SSIM
        print("-------------------4. Метод SSIM-------------------")
        start_time = time.time()
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        output_path = Path(f"{output_dir_path}/pred_ssim.txt")

        # Размер изображений
        IMAGE_SIZE = (64, 64)  # Чем ниже размер, тем быстрее считает

        # Предзагрузка изображений
        print("Загрузка тренировочных изображений...")
        train_images_data = ssm.load_images(train_images, IMAGE_SIZE)

        print("Загрузка тестовых изображений...")
        test_images_data = ssm.load_images(test_images, IMAGE_SIZE)

        preds = ssm.get_image_comparison(
            train_images=train_images_data,
            test_images=test_images_data,
            test_paths=test_images,
            threshold=0.2,  # Чем ниже порог, тем быстрее считает
        )
        end_time = time.time()
        ssim_preds_time = measure_time(start_time, end_time)

        metrics_ssim = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # ssim_preds = preds

        # Метод Histogram
        print("-------------------5. Метод Histogram-------------------")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_histogram.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )

        # Параметры гистограммы
        HIST_SIZE = 64  # Количество бинов на канал
        COMPARE_METHOD = "correl"  # Метрика сравнения (корреляция)

        # Предзагрузка гистограмм
        print("Расчет гистограмм для тренировочных изображений...")
        train_images_data = hgm.load_histograms(train_images, HIST_SIZE)

        print("Расчет гистограмм для тестовых изображений...")
        test_images_data = hgm.load_histograms(test_images, HIST_SIZE)

        preds = hgm.get_histogram_comparison(
            train_hists=train_images_data,
            test_hists=test_images_data,
            test_paths=test_images,
            threshold=0.85,
            method=COMPARE_METHOD,
        )
        end_time = time.time()
        hist_preds_time = measure_time(start_time, end_time)

        metrics_hist = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        hist_preds = preds

        # Метод ORB (Oriented FAST and Rotated BRIEF)
        print("-------------------6. Метод ORB-------------------")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_orb.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )

        # Параметры
        IMAGE_SIZE = (128, 128)

        # Предзагрузка изображений
        print("Загрузка тренировочных изображений...")
        train_images_data = om.load_images(train_images, target_size=IMAGE_SIZE)

        print("Загрузка тестовых изображений...")
        test_images_data = om.load_images(test_images, target_size=IMAGE_SIZE)

        preds = om.get_image_comparison(
            train_images=train_images_data,
            test_images=test_images_data,
            test_paths=test_images,
            threshold=40,
        )
        end_time = time.time()
        orb_preds_time = measure_time(start_time, end_time)

        metrics_orb = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        orb_preds = preds

        # Метод SIFT (Scale-Invariant Feature Transform)
        print("-------------------7. Метод SIFT-------------------")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_sift.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )

        # Предзагрузка дескрипторов SIFT
        print("Загрузка дескрипторов SIFT для тренировочных изображений...")
        train_images_data = sm.load_sift_descriptors(train_images)

        print("Загрузка дескрипторов SIFT для тестовых изображений...")
        test_images_data = sm.load_sift_descriptors(test_images)

        preds = sm.get_image_comparison(
            train_images=train_images_data,
            test_images=test_images_data,
            test_paths=test_images,
            threshold=10,
        )
        end_time = time.time()
        sift_preds_time = measure_time(start_time, end_time)

        metrics_orb = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # sift_preds = preds

        # ResNet50
        print("------------------8. ResNet50-------------------")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_resnet50.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        IMAGE_SIZE = (224, 224)
        embedder = TransformerEmbedder(IMAGE_SIZE, model_name="resnet50", device="cuda")

        # Создание эмбеддингов
        print("Создание эмбеддингов для тренировочных изображений...")
        train_embeddings = embedder.compute_embeddings(train_images)

        print("Создание эмбеддингов для тестовых изображений...")
        test_embeddings = embedder.compute_embeddings(test_images)

        preds = embedder.get_image_comparison(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_paths=test_images,
            threshold=0.75,
        )
        end_time = time.time()
        resnet_preds_time = measure_time(start_time, end_time)

        metrics_resnet = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        resnet_preds = preds

        # EfficientNet-B4
        print("------------------9. EfficientNet-B4-------------------")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_efficientnet_b4.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        IMAGE_SIZE = (380, 380)
        embedder = TransformerEmbedder(
            IMAGE_SIZE, model_name="efficientnet_b4", device="cuda"
        )

        # Создание эмбеддингов
        print("Создание эмбеддингов для тренировочных изображений...")
        train_embeddings = embedder.compute_embeddings(train_images)

        print("Создание эмбеддингов для тестовых изображений...")
        test_embeddings = embedder.compute_embeddings(test_images)

        preds = embedder.get_image_comparison(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_paths=test_images,
            threshold=0.95,
        )
        end_time = time.time()
        efficientnet_preds_time = measure_time(start_time, end_time)

        metrics_efficientnet_b4 = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # efficientnet_preds = preds

        # ConvNeXt-Tiny
        print("------------------10. ConvNeXt-Tiny-------------------")
        start_time = time.time()
        output_path = Path(f"{output_dir_path}/pred_convnext_tiny.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        IMAGE_SIZE = (224, 224)
        embedder = TransformerEmbedder(
            IMAGE_SIZE, model_name="convnext_tiny", device="cuda"
        )

        # Создание эмбеддингов
        print("Создание эмбеддингов для тренировочных изображений...")
        train_embeddings = embedder.compute_embeddings(train_images)

        print("Создание эмбеддингов для тестовых изображений...")
        test_embeddings = embedder.compute_embeddings(test_images)

        preds = embedder.get_image_comparison(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_paths=test_images,
            threshold=0.75,
        )
        end_time = time.time()
        convnext_preds_time = measure_time(start_time, end_time)

        metrics_convnext_tiny = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # convnext_preds = preds

        # Комбинации методов
        print("-------------------11. Комбинации методов (сложение предсказаний)-------------------")
        print("-------------------FAST_DHASH + HIST + ORB-------------------")
        combined_preds = fast_dhash_preds + hist_preds + orb_preds
        elapsed_time = fast_dhash_preds_time + hist_preds_time + orb_preds_time
        print(f"Время выполнения в миллисекундах: {elapsed_time:.5f}s ({int(elapsed_time * 1000)} ms)")
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------HIST + ORB-------------------")
        combined_preds = hist_preds + orb_preds
        elapsed_time = hist_preds_time + orb_preds_time
        print(f"Время выполнения в миллисекундах: {elapsed_time:.5f}s ({int(elapsed_time * 1000)} ms)")
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------FAST_DHASH + ORB-------------------")
        combined_preds = fast_dhash_preds + orb_preds
        elapsed_time = fast_dhash_preds_time + orb_preds_time
        print(f"Время выполнения в миллисекундах: {elapsed_time:.5f}s ({int(elapsed_time * 1000)} ms)")
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------ORB + ResNet-------------------")
        combined_preds = orb_preds + resnet_preds
        elapsed_time = orb_preds_time + resnet_preds_time
        print(f"Время выполнения в миллисекундах: {elapsed_time:.5f}s ({int(elapsed_time * 1000)} ms)")
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------HIST + ORB + ResNet-------------------")
        combined_preds = hist_preds + orb_preds + resnet_preds
        elapsed_time = hist_preds_time + orb_preds_time + resnet_preds_time
        print(f"Время выполнения в миллисекундах: {elapsed_time:.5f}s ({int(elapsed_time * 1000)} ms)")
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------12. Методы ORB (первоначальная обработка) --> ResNet (дополнительная обработка остальных изображений)-------------------")
        start_time = time.time()
        left_images = [img for img in test_images if img not in orb_preds]

        # ResNet50
        print("------------------ResNet50-------------------")
        
        output_path = Path(f"{output_dir_path}/pred_orb_resnet50.txt")
        train_images, test_images, leakage_images = open_files(
            train_images_path, test_images_path, image_extensions, leakage_images_path
        )
        IMAGE_SIZE = (512, 512)
        embedder = TransformerEmbedder(IMAGE_SIZE, model_name="resnet50", device="cuda")

        # Создание эмбеддингов
        print("Создание эмбеддингов для тренировочных изображений...")
        train_embeddings = embedder.compute_embeddings(train_images)

        print("Создание эмбеддингов для тестовых изображений...")
        test_embeddings = embedder.compute_embeddings(left_images)

        preds = embedder.get_image_comparison(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            test_paths=left_images,
            threshold=0.85,
        )
        end_time = time.time() + orb_preds_time
        orb_resnet_preds_time = measure_time(start_time, end_time)

        metrics_resnet = compute_metrics(leakage_images, orb_preds+preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")

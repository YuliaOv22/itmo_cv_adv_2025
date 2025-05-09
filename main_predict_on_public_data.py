from pathlib import Path
from metrics.compare_images_metrics import compute_metrics
from utils.read_save_files import read_images_from_directory, save_paths_to_file
import scripts.hash_methods as hm
import scripts.ssim_method as ssm
import scripts.histogram_method as hgm
import scripts.orb_method as om
import scripts.sift_method as sm
from scripts.cnn import TransformerEmbedder
from scripts.log import TeeLoggerContext


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

        # Метод pHash (Perceptual Hash)
        print("-------------------Метод pHash (Perceptual Hash)-------------------")
        output_path = Path(f"{output_dir_path}/pred_cached_pHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="phash"
        )
        metrics_phash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # phash_preds = preds

        # Метод dHash (Difference Hash)
        print("-------------------Метод dHash (Difference Hash)-------------------")
        output_path = Path(f"{output_dir_path}/pred_cached_dHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="dhash"
        )
        metrics_dhash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # dhash_preds = preds

        # Метод Fast dHash
        print("-------------------Метод Fast dHash-------------------")
        output_path = Path(f"{output_dir_path}/pred_cached_fast_dHash.txt")
        preds = hm.get_image_comparison(
            train_images, test_images, threshold=15, comparison_method="fast_dhash"
        )
        metrics_fast_dhash = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        fast_dhash_preds = preds

        # Метод SSIM
        print("-------------------Метод SSIM-------------------")
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

        metrics_ssim = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # ssim_preds = preds

        # Метод Histogram
        print("-------------------Метод Histogram-------------------")
        output_path = Path(f"{output_dir_path}/pred_histogram.txt")

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

        metrics_hist = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        hist_preds = preds

        # Метод ORB (Oriented FAST and Rotated BRIEF)
        print("-------------------Метод ORB-------------------")
        output_path = Path(f"{output_dir_path}/pred_orb.txt")

        # # Параметры
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

        metrics_orb = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        orb_preds = preds

        # Метод SIFT (Scale-Invariant Feature Transform)
        print("-------------------Метод SIFT-------------------")
        output_path = Path(f"{output_dir_path}/pred_sift.txt")

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

        metrics_orb = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # sift_preds = preds

        # ResNet50
        print("------------------ResNet50-------------------")
        output_path = Path(f"{output_dir_path}/pred_resnet50.txt")
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

        metrics_resnet = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        resnet_preds = preds

        # EfficientNet-B4
        print("------------------EfficientNet-B4-------------------")
        output_path = Path(f"{output_dir_path}/pred_efficientnet_b4.txt")
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

        metrics_efficientnet_b4 = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # efficientnet_preds = preds

        # ConvNeXt-Tiny
        print("------------------ConvNeXt-Tiny-------------------")
        output_path = Path(f"{output_dir_path}/pred_convnext_tiny.txt")
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

        metrics_convnext_tiny = compute_metrics(leakage_images, preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")
        # convnext_preds = preds

        # Комбинации методов
        print("-------------------Комбинации методов-------------------")
        print("-------------------Методы FAST_DHASH + HIST + ORB-------------------")
        combined_preds = fast_dhash_preds + hist_preds + orb_preds
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------Методы HIST + ORB-------------------")
        combined_preds = hist_preds + orb_preds
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------Методы FAST_DHASH + ORB-------------------")
        combined_preds = fast_dhash_preds + orb_preds
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------Методы ORB + ResNet-------------------")
        combined_preds = orb_preds + resnet_preds
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        print("-------------------Методы HIST + ORB + ResNet-------------------")
        combined_preds = hist_preds + orb_preds + resnet_preds
        metrics_combined = compute_metrics(leakage_images, combined_preds, test_images)

        
        print("-------------------Методы ORB --> ResNet-------------------")
        left_images = [img for img in test_images if img not in orb_preds]
        # ResNet50
        print("------------------ResNet50-------------------")
        output_path = Path(f"{output_dir_path}/pred_orb_resnet50.txt")
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

        metrics_resnet = compute_metrics(leakage_images, orb_preds+preds, test_images)
        save_paths_to_file(output_path, preds, format="txt")

import time
from pathlib import Path
from utils.log import TeeLoggerContext
import torch
import torchvision.models as models
from utils.read_save_files import (
    print_save_results_clustering,
)
from scripts.clustering_orb_resnet_kmeans_agg import (
    cluster_images as ci_resnet,
    extract_resnet_features_torch,
)
from utils.measure_time import measure_time


if __name__ == "__main__":

    log_path = Path("logs/clustering_on_data.log")
    log_path.parent.mkdir(exist_ok=True)

    with TeeLoggerContext(log_path):

        print("-------------------Private data-------------------")

        # Настройка переменных
        private_data_dir = Path("data")
        train_images_path_private = private_data_dir / "train"
        output_dir_path = Path("output")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        all_image_paths = [
            p
            for p in train_images_path_private.glob("*")
            if p.suffix.lower() in image_extensions
        ]

        output_path_result_txt_kmeans = "pred_groups_final_kmeans.txt"
        output_path_result_txt_agg = "pred_groups_final_agg.txt"

        # Параметры
        IMAGE_SIZE = (512, 512)
        N_VISUAL_WORDS = 50
        N_CLUSTERS = 52

        # Проверка доступности CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")

        print(
            "\n-------------------Кластеризация с помощью K-Means и ResNet-------------------"
        )
        start_time = time.time()

        # Загрузка предобученной модели ResNet-50
        resnet_model = models.resnet50(
            weights="ResNet50_Weights.IMAGENET1K_V1"
        )  # Лучшие веса для задачи
        resnet_model.fc = torch.nn.Identity()

        # Добавление слоя адаптивного усредненного пулинга
        feature_extractor = torch.nn.Sequential(
            *list(resnet_model.children())[:-1], torch.nn.AdaptiveAvgPool2d(1)
        ).to(device)

        # Извлечение признаков ResNet
        print(f"Извлечение признаков ResNet для {len(all_image_paths)} изображений...")
        resnet_features = extract_resnet_features_torch(
            all_image_paths, feature_extractor, IMAGE_SIZE, device
        )

        preds = ci_resnet(resnet_features, n_clusters=N_CLUSTERS, clust_method="kmeans")

        print_save_results_clustering(
            None,  # true_groups не известны для новых данных
            preds,
            output_dir_path,
            output_path_result_txt_kmeans,
        )

        end_time = time.time()
        agg_resnet_preds_time = measure_time(start_time, end_time)

        print(
            "\n-------------------Кластеризация с помощью Agglomerative Clustering и ResNet-------------------"
        )
        start_time = time.time()

        # Загрузка предобученной модели ResNet-50
        resnet_model = models.resnet50(
            weights="ResNet50_Weights.IMAGENET1K_V1"
        )  # Лучшие веса для задачи
        resnet_model.fc = torch.nn.Identity()

        # Добавление слоя адаптивного усредненного пулинга
        feature_extractor = torch.nn.Sequential(
            *list(resnet_model.children())[:-1], torch.nn.AdaptiveAvgPool2d(1)
        ).to(device)

        # Извлечение признаков ResNet
        print(f"Извлечение признаков ResNet для {len(all_image_paths)} изображений...")
        resnet_features = extract_resnet_features_torch(
            all_image_paths, feature_extractor, IMAGE_SIZE, device
        )

        preds = ci_resnet(
            resnet_features, n_clusters=N_CLUSTERS, clust_method="agg", linkage="ward"
        )

        print_save_results_clustering(
            None,  # true_groups не известны для новых данных
            preds,
            output_dir_path,
            output_path_result_txt_agg,
        )

        end_time = time.time()
        agg_resnet_preds_time = measure_time(start_time, end_time)

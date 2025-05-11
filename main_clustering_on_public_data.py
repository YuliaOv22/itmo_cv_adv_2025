import time
from pathlib import Path
from utils.log import TeeLoggerContext
import torch
import torchvision.models as models
import numpy as np
import scripts.detect_with_orb_method as om
from utils.read_save_files import (
    read_images_from_directory,
    print_save_results_clustering,
)
from scripts.clustering_orb_graph import group_images_with_orb_parallel
from scripts.clustering_orb_bovw_kmeans_agg import cluster_images as ci_bovw
from scripts.clustering_orb_resnet_kmeans_agg import (
    cluster_images as ci_resnet,
    extract_resnet_features_torch,
    extract_orb_features,
    create_combined_embeddings,
)


if __name__ == "__main__":

    log_path = Path("logs/clustering_on_public_data.log")
    log_path.parent.mkdir(exist_ok=True)

    with TeeLoggerContext(log_path):

        start_time = time.time()

        print("-------------------Public data-------------------")
        # Настройка переменных
        input_dir_path = Path("public_data")
        train_images_path = input_dir_path / "train"
        test_images_path = input_dir_path / "test"
        output_dir_path = Path("output")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        true_groups_path = input_dir_path / "groups.txt"

        # Чтение изображений из папок
        train_images, test_images = read_images_from_directory(
            train_images_path, test_images_path, image_extensions
        )
        all_image_paths = [
            p
            for p in train_images_path.glob("*")
            if p.suffix.lower() in image_extensions
        ]

        # Чтение файла с группами изображений
        with open(true_groups_path, "r") as f:
            lines = f.readlines()

        true_groups = [line.strip().split(",") for line in lines]
        print(f"Количество групп: {len(true_groups)}")
        print(f"Пример группы: {true_groups[0]}")

        IMAGE_SIZE = (128, 128)
        N_VISUAL_WORDS = 50
        N_CLUSTERS = len(true_groups)  # Количество истинных групп для сравнения

        print(
            "\n-------------------1. Метод ORB + Graph-based clustering-------------------"
        )
        output_path = Path(f"{output_dir_path}/clustering_orb.txt")

        # Предзагрузка изображений
        print("Загрузка тренировочных изображений...")
        train_images_data = om.load_images(train_images, target_size=IMAGE_SIZE)

        print("Группировка изображений с использованием ORB...")
        preds = group_images_with_orb_parallel(
            train_images, train_images_data, similarity_threshold=40
        )

        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_graph.txt",
        )
        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------Блок кластеризации с признаком ORB-------------------"
        )
        print(
            "\n-------------------2. Кластеризация с помощью K-Means и ORB (BoVW)-------------------"
        )
        start_time = time.time()
        preds = ci_bovw(
            all_image_paths,
            n_clusters=N_CLUSTERS,
            n_visual_words=N_VISUAL_WORDS,
            clust_method="kmeans",
        )
        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_bovw_kmeans_on_public_data.txt",
        )
        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------3. Кластеризация с помощью Agglomerative Clustering и ORB (BoVW)-------------------"
        )
        start_time = time.time()
        preds = ci_bovw(
            all_image_paths,
            n_clusters=N_CLUSTERS,
            n_visual_words=N_VISUAL_WORDS,
            clust_method="agg",
            linkage="ward",
        )

        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_bovw_agg_on_public_data.txt",
        )
        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------4. Кластеризация с помощью Agglomerative Clustering (average linkage) и ORB (BoVW)-------------------"
        )
        start_time = time.time()
        preds = ci_bovw(
            all_image_paths,
            n_clusters=N_CLUSTERS,
            n_visual_words=N_VISUAL_WORDS,
            clust_method="agg",
            linkage="average",
        )
        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_bovw_agg_avg_on_public_data.txt",
        )

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------Блок кластеризации с эмбеддингами ResNet-------------------"
        )
        # Параметры
        IMAGE_SIZE = (512, 512)
        ORB_FEATURE_SIZE = 32  # Размерность вектора признаков ORB
        N_CLUSTERS = len(true_groups)  # Количество истинных групп для сравнения
        # Проверка доступности CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")

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
        resnet_features = extract_resnet_features_torch(
            all_image_paths, feature_extractor, IMAGE_SIZE, device
        )

        print(
            "\n-------------------5. Кластеризация с помощью K-Means и ResNet-------------------"
        )

        preds = ci_resnet(
            resnet_features, n_clusters=N_CLUSTERS, clust_method="kmeans"
        )

        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_resnet_kmeans_on_public_data.txt",
        )

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------6. Кластеризация с помощью K-Means и ORB + ResNet-------------------"
        )
        start_time = time.time()

        # Извлечение признаков ORB
        orb_features = extract_orb_features(all_image_paths, ORB_FEATURE_SIZE)

        # Создание объединенных эмбеддингов
        combined_features = create_combined_embeddings(resnet_features, orb_features)

        combined_feature_vectors = np.array(list(combined_features.values()))
        if combined_feature_vectors.ndim != 2:
            raise ValueError(
                f"Ожидался двумерный массив объединенных признаков, получен массив размерности {combined_feature_vectors.ndim}"
            )
        preds = ci_resnet(
            combined_features, n_clusters=N_CLUSTERS, clust_method="kmeans"
        )

        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_resnet_kmeans_on_public_data.txt",
        )

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------7. Кластеризация с помощью Agglomerative Clustering и ResNet-------------------"
        )
        start_time = time.time()

        preds = ci_resnet(
            resnet_features, n_clusters=N_CLUSTERS, clust_method="agg", linkage="ward"
        )

        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_resnet_agg_on_public_data.txt",
        )

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

        print(
            "\n-------------------8. Кластеризация с помощью Agglomerative Clustering (average linkage) и ResNet-------------------"
        )
        start_time = time.time()

        preds = ci_resnet(
            resnet_features,
            n_clusters=N_CLUSTERS,
            clust_method="agg",
            linkage="average",
        )

        print_save_results_clustering(
            true_groups,
            preds,
            output_dir_path,
            "clustering_orb_resnet_agg_avg_on_public_data.txt",
        )

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Время выполнения: {final_time}")

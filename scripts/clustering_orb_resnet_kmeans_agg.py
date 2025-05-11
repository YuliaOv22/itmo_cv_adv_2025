from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering


class ImageDataset(Dataset):
    """Кастомный датасет для загрузки изображений."""

    def __init__(self, image_paths: list, transform: transforms.Compose = None) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, Path(img_path).name
        except Exception as e:
            print(f"Ошибка при загрузке {img_path}: {e}")
            return None, None


def extract_resnet_features_torch(
    image_paths: list, model: torch.nn.Module, image_size: tuple, device: torch.device
) -> dict:
    """Извлекает признаки из изображений с помощью ResNet-50."""

    model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    features_list = []
    names_list = []
    with torch.no_grad():
        for images, names in tqdm(dataloader, desc="Извлечение признаков ResNet-50"):
            if images is not None:
                images = images.to(device)
                outputs = model(images)
                # Убираем последние две размерности и переводим в NumPy
                batch_features = outputs.squeeze(-1).squeeze(-1).cpu().numpy()
                features_list.extend(batch_features)
                names_list.extend(list(names))

    return dict(zip(names_list, features_list))


def extract_orb_features(
    image_paths: list[Path], orb_feature_size: int = 32
) -> dict[str, np.ndarray]:
    """Извлекает признаки ORB из изображений."""

    orb = cv2.ORB_create()
    features = {}
    for img_path in tqdm(image_paths, desc="Извлечение признаков ORB"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                if des is not None:
                    # Используем только первые ORB_FEATURE_SIZE дескрипторов, дополняя нулями при необходимости
                    feature_vector = np.pad(
                        des.flatten(),
                        (
                            0,
                            max(0, orb_feature_size * 32 - des.shape[0] * des.shape[1]),
                        ),
                        "constant",
                    )[: orb_feature_size * 32]
                    features[Path(img_path).name] = feature_vector
                else:
                    features[Path(img_path).name] = np.zeros(
                        orb_feature_size * 32
                    )  # Если дескрипторы не найдены
            else:
                print(f"Ошибка при чтении изображения: {img_path}")
                features[Path(img_path).name] = np.zeros(orb_feature_size * 32)
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
            features[Path(img_path).name] = np.zeros(orb_feature_size * 32)
    return features


def create_combined_embeddings(resnet_features: dict, orb_features: dict) -> dict:
    """Объединяет признаки ResNet и ORB."""

    combined_embeddings = {}
    for name, resnet_feat in resnet_features.items():
        if name in orb_features:
            combined_embeddings[name] = np.concatenate(
                (resnet_feat, orb_features[name])
            )
        else:
            print(f"Предупреждение: отсутствуют признаки ORB для {name}")
            combined_embeddings[name] = (
                resnet_feat  # Использовать только ResNet, если нет ORB
            )
    return combined_embeddings


def cluster_images(
    features: dict, n_clusters: int, clust_method: str = "kmeans", linkage: str = "ward"
) -> list[list[str]]:
    """Кластеризует признаки с помощью K-Means или Agglomerative Clustering."""

    feature_vectors = np.array(list(features.values()))
    if clust_method == "kmeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if clust_method == "agg":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = clustering.fit_predict(feature_vectors)
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(list(features.keys())[i])
    return list(clusters.values())

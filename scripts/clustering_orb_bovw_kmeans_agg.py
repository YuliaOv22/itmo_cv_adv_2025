from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering


def extract_orb_features(image_path: Path) -> tuple:
    """Извлекает ORB ключевые точки и дескрипторы из изображения."""

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def create_bow_vocabulary(image_paths: list, n_visual_words: int) -> KMeans:
    """Создает словарь визуальных слов на основе ORB дескрипторов."""

    all_descriptors = []
    for path in tqdm(image_paths, desc="Извлечение дескрипторов для BoVW"):
        _, descriptors = extract_orb_features(path)
        if descriptors is not None:
            all_descriptors.extend(descriptors)

    if not all_descriptors:
        return None

    kmeans = KMeans(n_clusters=n_visual_words, random_state=42, n_init=10)
    kmeans.fit(np.array(all_descriptors))
    return kmeans


def create_bow_histogram(image_path: Path, vocabulary: KMeans) -> np.ndarray:
    """Создает гистограмму визуальных слов для заданного изображения."""

    _, descriptors = extract_orb_features(image_path)
    if descriptors is None or vocabulary is None:
        return None

    histogram = np.zeros(vocabulary.n_clusters)
    closest_words = vocabulary.predict(descriptors)
    for word_index in closest_words:
        histogram[word_index] += 1
    return histogram


def cluster_images(
    image_paths: list,
    n_clusters: int,
    n_visual_words: int,
    clust_method: str = "kmeans",
    linkage: str = "ward",
) -> dict:
    """Кластеризует изображения с помощью K-Means или Agglomerative Clustering на основе признаков BoVW (ORB)."""

    if n_clusters is None:
        raise ValueError("Необходимо указать количество кластеров (n_clusters).")

    vocabulary = create_bow_vocabulary(image_paths, n_visual_words)
    if vocabulary is None:
        return []

    image_features = {}
    for path in tqdm(image_paths, desc="Создание гистограмм BoVW"):
        histogram = create_bow_histogram(path, vocabulary)
        if histogram is not None:
            image_features[path.name] = histogram

    if not image_features:
        return []

    feature_vectors = np.array(list(image_features.values()))
    if clust_method == "kmeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if clust_method == "agg":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = clustering.fit_predict(feature_vectors)

    predicted_groups = defaultdict(list)
    image_names = list(image_features.keys())
    for i, label in enumerate(labels):
        predicted_groups[label].append(image_names[i])

    return list(predicted_groups.values())

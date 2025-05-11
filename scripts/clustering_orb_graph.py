from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.read_save_files import read_images_from_directory
import scripts.detect_with_orb_method as om
import time
from tqdm import tqdm
from metrics.clustering_metrics import (
    exact_match,
    partial_match,
)
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed


def compare_images(args: tuple) -> tuple:
    """Сравнивает два изображения с использованием ORB и возвращает результат сравнения."""
    i, j, image_names, image_data, similarity_threshold = args
    path1 = image_names[i]
    path2 = image_names[j]
    img1 = image_data[i]
    img2 = image_data[j]

    if om.orb_compare(img1, img2, threshold=similarity_threshold):
        return (path1, path2)
    return None


def group_images_with_orb_parallel(
    image_paths: list[Path],
    image_data: list,
    similarity_threshold: float,
    num_workers: int = 4,
):
    """Группирует изображения с использованием ORB, графов и параллельного выполнения."""
    image_names = [path.name for path in image_paths]
    graph = nx.Graph()
    graph.add_nodes_from(image_names)

    n_images = len(image_names)
    tasks = []
    for i in range(n_images):
        for j in range(
            i + 1, n_images
        ):  # Избегаем повторных сравнений и сравнения с самим собой
            tasks.append((i, j, image_names, image_data, similarity_threshold))

    edges = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compare_images, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Параллельное сравнение"
        ):
            result = future.result()
            if result:
                edges.append(result)

    graph.add_edges_from(edges)
    grouped_images = [list(group) for group in nx.connected_components(graph)]
    return grouped_images

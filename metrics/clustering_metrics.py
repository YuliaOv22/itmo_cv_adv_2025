def exact_match(true_groups: list, pred_clusters: list) -> float:
    """Считает долю предсказанных кластеров, которые точно совпадают с истинными группами."""

    # Преобразуем группы в множества для сравнения
    true_sets = [set(group) for group in true_groups]
    pred_sets = [set(cluster) for cluster in pred_clusters]

    # Считаем, сколько предсказанных кластеров есть в истинных группах
    print(f"Совпавшие группы: ")
    correct = 0
    for p in pred_sets:
        if p in true_sets:
            correct += 1
            print(p)
    return correct / len(true_sets)


def partial_match(true_groups: list, pred_clusters: list, threshold: float = 0.8):
    """Считает долю предсказанных кластеров, которые частично совпадают с истинными группами."""

    # Преобразуем группы в множества для сравнения
    true_sets = [set(group) for group in true_groups]
    correct = 0

    for p in pred_clusters:
        p_set = set(p)
        max_overlap = max(len(p_set & t) / len(p_set) for t in true_sets)
        if max_overlap >= threshold:
            correct += 1

    return correct / len(true_sets)

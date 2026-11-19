"""Кодировки структуры графа на уровне узлов (класс A таксономии).

Всё считается на numpy/scipy один раз при подготовке датасета (``pre_transform`` в PyG),
поэтому torch здесь не нужен, а функции можно тестировать без GPU.

Обозначения (как в пособии, гл. 9): граф G = (V, A), n = |V|, A — матрица смежности,
D — диагональная матрица степеней, L = E - D^{-1/2} A D^{-1/2} — нормированный лапласиан.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def adjacency_from_edge_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Плотная симметричная матрица смежности (0/1) из списка рёбер формы [2, m].

    Граф считается неориентированным: ребро (i, j) добавляется в обе стороны.
    Кратные рёбра и петли игнорируются (значение 1, диагональ обнуляется).
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.size:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """L = E - D^{-1/2} A D^{-1/2}. Для изолированных узлов (степень 0) берём D^{-1/2} = 0."""
    deg = adj.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = deg[nz] ** -0.5
    return np.eye(adj.shape[0]) - (inv_sqrt[:, None] * adj * inv_sqrt[None, :])


def laplacian_pe(adj: np.ndarray, k: int) -> np.ndarray:
    """LapPE: k собственных векторов нормированного лапласиана с наименьшими
    нетривиальными собственными значениями (Dwivedi & Bresson, 2020).

    Возвращает массив [n, k]. Нулевые моды всех компонент связности пропускаются.
    Если у графа меньше k + 1 узлов, недостающие столбцы заполняются нулями.
    Знак собственных векторов произволен — при обучении применяется случайная смена знака
    (см. ``models.GraphTransformer``).
    """
    n = adj.shape[0]
    lap = normalized_laplacian(adj)
    eigvals, eigvecs = scipy.linalg.eigh(lap)      # отсортированы по возрастанию λ
    eigvecs = eigvecs[:, eigvals > 1e-8]
    pe = np.zeros((n, k), dtype=np.float32)
    take = min(k, eigvecs.shape[1])
    pe[:, :take] = eigvecs[:, :take]
    return pe


def random_walk_se(adj: np.ndarray, k: int) -> np.ndarray:
    """RWSE: вероятности возврата в узел за 1..k шагов случайного блуждания
    (Dwivedi et al., 2022; используется в GraphGPS).

    P = D^{-1} A — матрица перехода; p_i = (P_ii, (P^2)_ii, ..., (P^k)_ii). Возвращает [n, k].
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    P = inv_deg[:, None] * adj
    se = np.zeros((n, k), dtype=np.float32)
    Pk = np.eye(n)
    for step in range(k):
        Pk = Pk @ P
        se[:, step] = np.diag(Pk)
    return se


def shortest_path_matrix(adj: np.ndarray, max_dist: int) -> np.ndarray:
    """Длины кратчайших путей между всеми парами (BFS через степени матрицы смежности).

    Эталонная numpy-реализация для тестов; в обучении используется torch-версия
    ``pair_encodings.shortest_path_dense`` (та же логика на GPU).
    Конечные расстояния ограничиваются max_dist; max_dist + 1 — отдельный индекс несвязных пар.
    """
    n = adj.shape[0]
    dist = np.full((n, n), max_dist + 1, dtype=np.int64)
    reached = np.eye(n, dtype=bool)
    frontier = np.eye(n, dtype=bool)
    dist[reached] = 0
    for d in range(1, n):
        frontier = ((frontier.astype(np.float64) @ adj) > 0) & ~reached
        dist[frontier] = min(d, max_dist)
        reached |= frontier
        if not frontier.any():
            break
    return dist


def rrwp(adj: np.ndarray, k: int) -> np.ndarray:
    """RRWP: вероятности случайного блуждания из i в j за 0..k-1 шагов (Ma et al., GRIT, 2023).

    Возвращает [n, n, k]; срез [:, :, 0] — единичная матрица, диагональ по k совпадает с RWSE
    (со сдвигом на один шаг). Эталонная numpy-реализация для тестов.
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    P = inv_deg[:, None] * adj
    out = np.zeros((n, n, k), dtype=np.float32)
    Pk = np.eye(n)
    for step in range(k):
        out[:, :, step] = Pk
        Pk = Pk @ P
    return out


def degrees(adj: np.ndarray, max_degree: int = 64) -> np.ndarray:
    """Степени узлов, ограниченные сверху max_degree (для таблицы эмбеддингов)."""
    return np.minimum(adj.sum(axis=1).astype(np.int64), max_degree)

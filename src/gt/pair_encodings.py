"""Кодировки структуры на уровне пар узлов (класс B таксономии), вычисляемые на GPU «на лету».

Для задач на уровне графа графы маленькие (ZINC ~23 узла, Peptides ~150), поэтому
батч представляется плотно: матрица смежности [B, N, N]. Степени этой матрицы дают
и кратчайшие расстояния (SPD, Graphormer), и вероятности случайных блужданий (RRWP, GRIT)
за несколько батчевых матричных умножений — это дешевле, чем хранить n^2 чисел на граф в датасете.
"""

from __future__ import annotations

import torch
from torch import Tensor


def shortest_path_dense(adj: Tensor, max_dist: int) -> Tensor:
    """Длины кратчайших путей для батча плотных матриц смежности.

    adj: [B, N, N] (0/1, float). Возвращает long-тензор [B, N, N]:
    0 на диагонали, min(d, max_dist) для связных пар, max_dist + 1 — «недостижимо».
    Алгоритм — BFS «фронтами»: фронт расстояния d — это соседи фронта d-1, ещё не достигнутые.
    """
    B, N, _ = adj.shape
    eye = torch.eye(N, dtype=torch.bool, device=adj.device).unsqueeze(0).expand(B, N, N)
    dist = torch.full((B, N, N), max_dist + 1, dtype=torch.long, device=adj.device)
    dist = dist.masked_fill(eye, 0)
    reached = eye.clone()
    frontier = eye.clone()
    for d in range(1, N):
        frontier = (torch.bmm(frontier.float(), adj) > 0) & ~reached
        if not frontier.any():
            break
        dist = dist.masked_fill(frontier, min(d, max_dist))
        reached = reached | frontier
    return dist


def rrwp_dense(adj: Tensor, k: int) -> Tensor:
    """RRWP для батча: [B, N, N, k], срез по последней оси — P^0, P^1, ..., P^{k-1}, P = D^{-1} A."""
    deg = adj.sum(dim=-1, keepdim=True)
    P = adj / deg.clamp(min=1.0)
    B, N, _ = adj.shape
    Pk = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, N, N)
    feats = []
    for _ in range(k):
        feats.append(Pk)
        Pk = torch.bmm(Pk, P)
    return torch.stack(feats, dim=-1)

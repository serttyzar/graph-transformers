"""Загрузка датасетов и добавление кодировок структуры.

Для задач на уровне графа кодировки узлов (LapPE, RWSE) считаются один раз при загрузке
(numpy, ``encodings.py``) и кэшируются на диск, чтобы не пересчитывать для каждого сида/варианта.
Кодировки пар (SPD, RRWP) в датасете не хранятся — их считает модель на GPU.

Единый формат ``info``:
  node_cats: список числа категорий по столбцам x (категориальные признаки) или None;
  node_dim:  размерность вещественных признаков (если node_cats is None);
  edge_cats: аналогично для edge_attr или None (нет признаков рёбер);
  out_dim:   размер выхода; task: 'regression' | 'classification' | 'multilabel';
  metric:    'mae' | 'acc' | 'ap'.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from . import encodings as enc



def infer_cats(graphs: list[Data], key: str) -> list[int] | None:
    """Число категорий по каждому столбцу целочисленного признака (max + 1) по всему списку графов."""
    vals = [getattr(g, key) for g in graphs if getattr(g, key, None) is not None]
    if not vals:
        return None
    t = torch.cat([v.view(v.size(0), -1) for v in vals], dim=0)
    return (t.max(dim=0).values + 1).tolist()


def add_encodings(data: Data, lappe_k: int, rwse_k: int) -> Data:
    """Добавляет data.lappe [n, lappe_k] и data.rwse [n, rwse_k] (если k > 0)."""
    if lappe_k == 0 and rwse_k == 0:
        return data
    n = int(data.num_nodes)
    adj = enc.adjacency_from_edge_index(data.edge_index.numpy(), n)
    if lappe_k > 0:
        data.lappe = torch.from_numpy(enc.laplacian_pe(adj, lappe_k))
    if rwse_k > 0:
        data.rwse = torch.from_numpy(enc.random_walk_se(adj, rwse_k))
    return data


def _with_encodings(dataset, cache_path: str, lappe_k: int, rwse_k: int) -> list[Data]:
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    out = [add_encodings(d, lappe_k, rwse_k) for d in dataset]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".pt.tmp", dir=os.path.dirname(cache_path))
    os.close(fd)
    try:
        torch.save(out, temporary)
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return out


def load_graph_dataset(name: str, root: str, lappe_k: int = 0, rwse_k: int = 0,
                       max_graphs: int | None = None):
    """Возвращает (splits, info): splits = {'train': [...], 'val': [...], 'test': [...]} или
    {'all': [...]} для CSL (фолды строятся функцией ``csl_folds``)."""
    tag = f"v2_lappe{lappe_k}_rwse{rwse_k}"
    if max_graphs is not None and name != "csl":
        tag += f"_limit{max_graphs}"
    if name == "zinc":
        from torch_geometric.datasets import ZINC
        splits = {}
        for split in ("train", "val", "test"):
            ds = ZINC(os.path.join(root, "ZINC"), subset=True, split=split)
            if max_graphs is not None:
                ds = ds[:max_graphs]
            splits[split] = _with_encodings(ds, os.path.join(root, "enc_cache", f"zinc_{split}_{tag}.pt"),
                                            lappe_k, rwse_k)
        all_graphs = splits["train"] + splits["val"] + splits["test"]
        info = dict(node_cats=infer_cats(all_graphs, "x"), edge_cats=infer_cats(all_graphs, "edge_attr"),
                    out_dim=1, task="regression", metric="mae")      # ZINC: 28 типов атомов, 4 типа связей
        return splits, info

    if name == "csl":
        from torch_geometric.datasets import GNNBenchmarkDataset
        ds = GNNBenchmarkDataset(os.path.join(root, "CSL"), name="CSL")
        graphs = []
        for d in ds:                                    # у CSL нет признаков — ставим константу
            d = Data(x=torch.zeros(d.num_nodes, dtype=torch.long), edge_index=d.edge_index, y=d.y)
            graphs.append(d)
        graphs = _with_encodings(graphs, os.path.join(root, "enc_cache", f"csl_{tag}.pt"), lappe_k, rwse_k)
        info = dict(node_cats=[1], edge_cats=None, out_dim=10, task="classification", metric="acc")
        return {"all": graphs}, info

    if name == "peptides-func":
        from torch_geometric.datasets import LRGBDataset
        splits = {}
        for split in ("train", "val", "test"):
            ds = LRGBDataset(os.path.join(root, "LRGB"), name="Peptides-func", split=split)
            if max_graphs is not None:
                ds = ds[:max_graphs]
            splits[split] = _with_encodings(ds, os.path.join(root, "enc_cache", f"pepfunc_{split}_{tag}.pt"),
                                            lappe_k, rwse_k)
        all_graphs = splits["train"] + splits["val"] + splits["test"]
        info = dict(node_cats=infer_cats(all_graphs, "x"), edge_cats=infer_cats(all_graphs, "edge_attr"),
                    out_dim=10, task="multilabel", metric="ap")        # OGB atom/bond features
        return splits, info

    raise ValueError(f"неизвестный датасет: {name}")


def csl_folds(graphs: list[Data], fold: int, num_folds: int = 5, seed: int = 0):
    """Стратифицированное разбиение CSL на train/val/test: тест — фолд ``fold``,
    валидация — следующий фолд, обучение — остальные. Фолды фиксированы (seed=0) для всех запусков."""
    if num_folds < 3 or not 0 <= fold < num_folds:
        raise ValueError("CSL: нужны >= 3 фолда и 0 <= fold < num_folds")
    y = np.array([int(g.y) for g in graphs])
    rng = np.random.RandomState(seed)
    fold_id = np.empty(len(graphs), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold_id[idx] = np.arange(len(idx)) % num_folds
    test_f, val_f = fold, (fold + 1) % num_folds
    return {
        "train": [g for g, f in zip(graphs, fold_id) if f not in (test_f, val_f)],
        "val": [g for g, f in zip(graphs, fold_id) if f == val_f],
        "test": [g for g, f in zip(graphs, fold_id) if f == test_f],
    }


def load_ogbn_arxiv(root: str):
    """ogbn-arxiv: x [N,128] float, y [N], edge_index (делается неориентированным в моделях),
    split_idx = {'train','valid','test'}."""
    mirror = Path(root) / 'OGB_mirror' / 'ogbn_arxiv.npz'
    if mirror.exists():
        with np.load(mirror, allow_pickle=False) as saved:
            data = Data(x=torch.from_numpy(saved['x']), y=torch.from_numpy(saved['y']).long(),
                        edge_index=torch.from_numpy(saved['edge_index']).long())
            split = {k: torch.from_numpy(saved[k]).long() for k in ('train', 'valid', 'test')}
        provenance = json.loads(mirror.with_name('provenance.json').read_text())
        info = dict(node_dim=128, out_dim=40, task='classification', metric='acc', data_provenance=provenance)
        return data, split, info
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset("ogbn-arxiv", root=os.path.join(root, "OGB"))
    data = ds[0]
    data.y = data.y.view(-1)
    split = ds.get_idx_split()
    info = dict(node_dim=data.x.size(1), out_dim=int(ds.num_classes), task="classification", metric="acc")
    return data, split, info


def synthetic_graph(num_nodes: int, avg_degree: int = 10, feat_dim: int = 64, seed: int = 0) -> Data:
    """Случайный граф (модель Эрдёша–Реньи по числу рёбер) для микробенчмарка масштабирования."""
    g = torch.Generator().manual_seed(seed)
    m = num_nodes * avg_degree // 2
    src = torch.randint(0, num_nodes, (m,), generator=g)
    dst = torch.randint(0, num_nodes, (m,), generator=g)
    keep = src != dst
    ei = torch.stack([torch.cat([src[keep], dst[keep]]), torch.cat([dst[keep], src[keep]])])
    x = torch.randn(num_nodes, feat_dim, generator=g)
    return Data(x=x, edge_index=ei, num_nodes=num_nodes)

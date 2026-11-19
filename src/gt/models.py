"""Модели для задач на уровне графа (ZINC, CSL, Peptides-func).

Три архитектуры с общими входными блоками:
- ``GraphTransformer`` — трансформер с полным вниманием внутри графа; способ внедрения структуры
  задаётся конфигом (класс A — ``node_enc``; класс B — ``pair_enc`` / ``edge_bias``; класс C — ``attn_mask``);
- ``MPNN``            — стек слоёв GINE (при отсутствии признаков рёбер вырождается в GIN);
- ``GPS``             — гибрид: слой GINE и глобальное внимание параллельно.

Конфиг модели (словарь, см. configs/*.yaml):
  model: gt | mpnn | gps
  node_enc: список из {lappe, rwse, degree}
  pair_enc: null | spd | rrwp
  edge_bias: bool            (признаки рёбер как bias внимания, только gt)
  attn_mask: full | neighbors (только gt)
  lappe_k, lappe_sign_flip, rwse_k, rrwp_k, spd_max, max_degree
  layers, heads, d_model, d_ff, dropout, attn_dropout
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch

from .layers import GINELayer, GPSLayer, TransformerLayer
from .pair_encodings import rrwp_dense, shortest_path_dense


# ---------------------------------------------------------------------------
# Входные блоки
# ---------------------------------------------------------------------------
class FeatureEncoder(nn.Module):
    """Признаки узлов/рёбер -> вектор размерности d.

    Категориальные признаки (long, несколько столбцов) — сумма эмбеддингов по столбцам
    (как AtomEncoder/BondEncoder в OGB); вещественные — линейный слой.
    ``num_cats`` — список числа категорий по столбцам, либо None для вещественных признаков
    (тогда нужен ``in_dim``).
    """

    def __init__(self, d_model: int, num_cats: list[int] | None, in_dim: int | None = None):
        super().__init__()
        self.categorical = num_cats is not None
        if self.categorical:
            self.embs = nn.ModuleList(nn.Embedding(n, d_model) for n in num_cats)
        else:
            self.lin = nn.Linear(in_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if self.categorical:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            return sum(emb(x[:, i]) for i, emb in enumerate(self.embs))
        return self.lin(x.float())


class StructuralEncoder(nn.Module):
    """Кодировки класса A: сумма проекций LapPE, RWSE и эмбеддинга степени (что включено в конфиге).

    LapPE при обучении подвергается случайной смене знака по каждому собственному вектору
    и каждому графу (борьба с неоднозначностью знака).
    """

    def __init__(self, cfg: dict, d_model: int):
        super().__init__()
        self.use = set(cfg.get("node_enc") or [])
        self.sign_flip = bool(cfg.get("lappe_sign_flip", True))
        self.raw_norms = nn.ModuleDict()
        if cfg.get('raw_pe_norm') == 'batch':
            for name, width in [('lappe', cfg['lappe_k']), ('rwse', cfg['rwse_k'])]:
                if name in self.use:
                    self.raw_norms[name] = nn.BatchNorm1d(width)
        if "lappe" in self.use:
            self.lappe_lin = nn.Linear(cfg["lappe_k"], d_model)
        if "rwse" in self.use:
            self.rwse_lin = nn.Linear(cfg["rwse_k"], d_model)
        if "degree" in self.use:
            self.max_degree = int(cfg.get("max_degree", 64))
            self.deg_emb = nn.Embedding(self.max_degree + 1, d_model)

    def forward(self, data) -> Tensor | None:
        out = None
        if "lappe" in self.use:
            pe = data.lappe
            if self.training and self.sign_flip:
                num_graphs = int(data.batch.max()) + 1
                sign = torch.randint(0, 2, (num_graphs, pe.size(1)), device=pe.device) * 2 - 1
                pe = pe * sign[data.batch]
            if 'lappe' in self.raw_norms:
                pe = self.raw_norms['lappe'](pe)
            out = self.lappe_lin(pe)
        if "rwse" in self.use:
            rwse = self.raw_norms['rwse'](data.rwse) if 'rwse' in self.raw_norms else data.rwse
            t = self.rwse_lin(rwse)
            out = t if out is None else out + t
        if "degree" in self.use:
            deg = degree(data.edge_index[1], data.num_nodes).long().clamp(max=self.max_degree)
            t = self.deg_emb(deg)
            out = t if out is None else out + t
        return out


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_model // 2, out_dim))

    def forward(self, g: Tensor) -> Tensor:
        return self.net(g)


# ---------------------------------------------------------------------------
# Графовый трансформер
# ---------------------------------------------------------------------------
class GraphTransformer(nn.Module):
    def __init__(self, cfg: dict, info: dict):
        super().__init__()
        d, H = cfg["d_model"], cfg["heads"]
        self.cfg = cfg
        self.pooling = cfg.get('pooling', 'mean')
        if self.pooling not in ('mean', 'sum'):
            raise ValueError('pooling must be mean or sum')
        self.node_enc = FeatureEncoder(d, info["node_cats"], info.get("node_dim"))
        self.struct_enc = StructuralEncoder(cfg, d)

        self.pair_enc = cfg.get("pair_enc")
        if self.pair_enc == "spd":
            self.spd_max = int(cfg["spd_max"])
            self.spd_emb = nn.Embedding(self.spd_max + 2, H)          # 0..spd_max и «недостижимо»
        elif self.pair_enc == "rrwp":
            self.rrwp_k = int(cfg["rrwp_k"])
            self.rrwp_mlp = nn.Sequential(nn.Linear(self.rrwp_k, 32), nn.ReLU(), nn.Linear(32, H))
        elif self.pair_enc not in (None, "none"):
            raise ValueError(f"неизвестный pair_enc: {self.pair_enc}")

        self.edge_bias = bool(cfg.get("edge_bias", False))
        if self.edge_bias:
            assert info["edge_cats"] is not None, "edge_bias требует признаков рёбер"
            self.edge_enc = FeatureEncoder(d, info["edge_cats"])
            self.edge_bias_lin = nn.Linear(d, H, bias=False)           # не-рёбра дают ровно 0

        self.attn_mask = cfg.get("attn_mask", "full")
        assert self.attn_mask in ("full", "neighbors")

        self.layers = nn.ModuleList(
            TransformerLayer(d, H, cfg["d_ff"], cfg["dropout"], cfg["attn_dropout"])
            for _ in range(cfg["layers"]))
        self.ln_out = nn.LayerNorm(d)
        self.head = PredictionHead(d, info["out_dim"], cfg["dropout"])

    def attention_inputs(self, data, N: int):
        """bias [B,H,N,N] и pair_mask [B,N,N] по конфигу (None, если не используются)."""
        if self.pair_enc in (None, "none") and not self.edge_bias and self.attn_mask == "full":
            return None, None
        adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=N).clamp(max=1.0)
        bias = None
        if self.pair_enc == "spd":
            dist = shortest_path_dense(adj, self.spd_max)
            bias = self.spd_emb(dist).permute(0, 3, 1, 2)
        elif self.pair_enc == "rrwp":
            bias = self.rrwp_mlp(rrwp_dense(adj, self.rrwp_k)).permute(0, 3, 1, 2)
        if self.edge_bias:
            e = self.edge_enc(data.edge_attr)
            # bias=False: проекция коммутирует с суммой дубликатов в to_dense_adj.
            # Уплотняем H голов вместо d признаков на пару (Peptides: 4 вместо 100).
            e = self.edge_bias_lin(e)
            eb = to_dense_adj(data.edge_index, data.batch, edge_attr=e, max_num_nodes=N).permute(0, 3, 1, 2)
            bias = eb if bias is None else bias + eb
        pair_mask = None
        if self.attn_mask == "neighbors":
            eye = torch.eye(N, dtype=torch.bool, device=adj.device)
            pair_mask = (adj > 0) | eye.unsqueeze(0)
        return bias, pair_mask

    def forward(self, data) -> Tensor:
        x = self.node_enc(data.x)
        s = self.struct_enc(data)
        if s is not None:
            x = x + s
        h, node_mask = to_dense_batch(x, data.batch)
        N = h.size(1)
        bias, pair_mask = self.attention_inputs(data, N)
        for layer in self.layers:
            h = layer(h, node_mask, bias, pair_mask)
        h = self.ln_out(h)
        m = node_mask.unsqueeze(-1).to(h.dtype)
        g = (h * m).sum(1)
        if self.pooling == 'mean':
            g = g / m.sum(1).clamp(min=1.0)
        return self.head(g)


# ---------------------------------------------------------------------------
# MPNN (GINE / GIN)
# ---------------------------------------------------------------------------
class MPNN(nn.Module):
    def __init__(self, cfg: dict, info: dict):
        super().__init__()
        d = cfg["d_model"]
        self.pooling = cfg.get('pooling', 'mean')
        if self.pooling not in ('mean', 'sum'):
            raise ValueError('pooling must be mean or sum')
        self.node_enc = FeatureEncoder(d, info["node_cats"], info.get("node_dim"))
        self.struct_enc = StructuralEncoder(cfg, d)
        self.has_edge_attr = bool(cfg.get("use_edge_attr", True)) and info["edge_cats"] is not None
        if self.has_edge_attr:
            self.edge_enc = FeatureEncoder(d, info["edge_cats"])
        self.layers = nn.ModuleList(
            GINELayer(d, cfg["d_ff"], cfg["dropout"], use_edge_attr=self.has_edge_attr)
            for _ in range(cfg["layers"]))
        self.head = PredictionHead(d, info["out_dim"], cfg["dropout"])

    def edge_features(self, data) -> Tensor | None:
        if self.has_edge_attr:
            return self.edge_enc(data.edge_attr)
        return None

    def forward(self, data) -> Tensor:
        x = self.node_enc(data.x)
        s = self.struct_enc(data)
        if s is not None:
            x = x + s
        e = self.edge_features(data)
        for layer in self.layers:
            x = layer(x, data.edge_index, e)
        pool = global_add_pool if self.pooling == 'sum' else global_mean_pool
        return self.head(pool(x, data.batch))


# ---------------------------------------------------------------------------
# GPS (гибрид)
# ---------------------------------------------------------------------------
class GPS(MPNN):
    def __init__(self, cfg: dict, info: dict):
        super().__init__(cfg, info)
        d, H = cfg["d_model"], cfg["heads"]
        self.layers = nn.ModuleList(
            GPSLayer(d, H, cfg["d_ff"], cfg["dropout"], cfg["attn_dropout"],
                     use_edge_attr=self.has_edge_attr) for _ in range(cfg["layers"]))

    def forward(self, data) -> Tensor:
        x = self.node_enc(data.x)
        s = self.struct_enc(data)
        if s is not None:
            x = x + s
        e = self.edge_features(data)
        # Функции перехода разреженное <-> плотное представление для глобального внимания.
        _, node_mask = to_dense_batch(x, data.batch)

        def to_dense(t):
            return to_dense_batch(t, data.batch)[0]

        def from_dense(h):
            return h[node_mask]

        for layer in self.layers:
            x = layer(x, data.edge_index, e, to_dense, from_dense, node_mask)
        pool = global_add_pool if self.pooling == 'sum' else global_mean_pool
        return self.head(pool(x, data.batch))


# ---------------------------------------------------------------------------
# Сборка и подбор ширины под бюджет параметров
# ---------------------------------------------------------------------------
MODELS = {"gt": GraphTransformer, "mpnn": MPNN, "gps": GPS}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, info: dict) -> nn.Module:
    return MODELS[cfg["model"]](cfg, info)


def fit_budget(cfg: dict, info: dict, budget: int, tol: float = 0.05) -> dict:
    """Подбирает d_model (кратно числу голов) и d_ff так, чтобы число параметров попало в
    budget ± tol. Возвращает копию конфига с заполненными d_model и d_ff.

    Шаг 1: наибольшее d_model, при котором модель с d_ff = 2·d_model не превышает бюджет.
    Шаг 2: d_ff подбирается линейной интерполяцией (число параметров линейно по d_ff).
    """
    cfg = copy.deepcopy(cfg)
    heads = int(cfg.get("heads", 4))
    if cfg.get("d_model") and cfg.get("d_ff"):
        return cfg                                                     # размеры заданы явно

    def n_params(d_model: int, d_ff: int) -> int:
        c = dict(cfg, d_model=d_model, d_ff=d_ff)
        return count_params(build_model(c, info))

    best_d = heads
    for d in range(heads, 1024, heads):
        if n_params(d, 2 * d) <= budget:
            best_d = d
        else:
            break
    p0, p1 = n_params(best_d, 2 * best_d), n_params(best_d, 2 * best_d + 64)
    slope = (p1 - p0) / 64.0
    d_ff = int(round(2 * best_d + (budget - p0) / max(slope, 1e-9)))
    d_ff = max(d_ff, best_d)
    cfg["d_model"], cfg["d_ff"] = best_d, d_ff
    total = n_params(best_d, d_ff)
    if abs(total - budget) > tol * budget:
        raise RuntimeError(f"не удалось уложиться в бюджет: {total} vs {budget} (d={best_d}, d_ff={d_ff})")
    return cfg

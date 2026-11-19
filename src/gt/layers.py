"""Слои. Три места, куда в трансформер вводится структура графа, видны прямо в сигнатуре
``DenseAttention.forward``:

- ``bias``      — добавка к логитам внимания по парам узлов (класс B: SPD, RRWP, признаки рёбер);
- ``pair_mask`` — какие пары вообще могут взаимодействовать (класс C: внимание по соседям);
- сами признаки узлов уже содержат кодировки класса A (LapPE, RWSE) — они добавляются в модели.

Остальные слои: разреженное внимание по списку рёбер (для больших графов и Exphormer-подобного
шаблона), линейное внимание (SGFormer), GINE (MPNN) и гибридный слой GPS.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.utils import softmax as scatter_softmax


# ---------------------------------------------------------------------------
# Полное внимание внутри графа (плотный батч)
# ---------------------------------------------------------------------------
class DenseAttention(nn.Module):
    """Многоголовое внимание по плотному батчу графов.

    h: [B, N, d]; node_mask: [B, N] (True — настоящий узел, False — паддинг);
    bias: [B, H, N, N] или None; pair_mask: [B, N, N] (True — пара разрешена) или None.
    """

    def __init__(self, d_model: int, heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0, "d_model должно делиться на число голов"
        self.heads = heads
        self.d_head = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = attn_dropout

    def forward(self, h: Tensor, node_mask: Tensor, bias: Tensor | None = None,
                pair_mask: Tensor | None = None) -> Tensor:
        B, N, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # [B, N, d] -> [B, H, N, d_head]
        q = q.view(B, N, self.heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)  # [B, H, N, N]
        if bias is not None:
            scores = scores + bias
        allowed = node_mask[:, None, None, :]                       # ключи-паддинг запрещены
        if pair_mask is not None:
            allowed = allowed & pair_mask[:, None, :, :]
        scores = scores.masked_fill(~allowed, float("-inf"))
        # У полностью закрытой строки softmax(-inf, ...) имеет NaN и в backward.
        has_keys = allowed.any(dim=-1, keepdim=True)
        scores = torch.where(has_keys, scores, torch.zeros_like(scores))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~allowed, 0.0)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
        return self.out(out) * node_mask.unsqueeze(-1).to(out.dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class TransformerLayer(nn.Module):
    """Pre-LN слой: h + Attn(LN(h)) -> h + FFN(LN(h))."""

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = DenseAttention(d_model, heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, node_mask: Tensor, bias: Tensor | None = None,
                pair_mask: Tensor | None = None) -> Tensor:
        h = h + self.dropout(self.attn(self.ln1(h), node_mask, bias, pair_mask))
        h = h + self.dropout(self.ff(self.ln2(h)))
        return h


# ---------------------------------------------------------------------------
# Разреженное внимание по списку рёбер (для больших графов; Exphormer-подобный шаблон)
# ---------------------------------------------------------------------------
class SparseAttention(nn.Module):
    """Внимание только по парам из ``edge_index`` (src -> dst): для узла dst softmax по входящим рёбрам.

    Память сообщений O(|E| * d), логитов O(|E| * H). Список рёбер может включать исходные рёбра, рёбра экспандера
    и рёбра к виртуальным узлам — модуль об этом не знает (см. ``models_node.expander_edges``).
    Петли (i -> i) нужно добавить заранее, иначе изолированный узел не получит ничего.
    """

    def __init__(self, d_model: int, heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0
        self.heads, self.d_head = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = attn_dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        N = x.size(0)
        src, dst = edge_index
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(N, self.heads, self.d_head)
        k = k.view(N, self.heads, self.d_head)
        v = v.view(N, self.heads, self.d_head)
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.d_head)     # [E, H]
        alpha = scatter_softmax(score, dst, num_nodes=N)                 # softmax по входящим рёбрам
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)
        msg = v[src] * alpha.unsqueeze(-1)                               # [E, H, d_head]
        out = torch.zeros(N, self.heads, self.d_head, device=x.device, dtype=msg.dtype)
        out.index_add_(0, dst, msg)
        return self.out(out.view(N, -1))


class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SparseAttention(d_model, heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), edge_index))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Линейное внимание (SGFormer, Wu et al., 2023)
# ---------------------------------------------------------------------------
class LinearAttention(nn.Module):
    """Один слой «простого» линейного глобального внимания по всем N узлам за O(N d^2).

    Q~ = Q / ||Q||_F, K~ = K / ||K||_F;
    Z = D^{-1} [ V + (1/N) Q~ (K~^T V) ],  D_ii = 1 + (1/N) q~_i · (sum_j k~_j),
    Это учебная SGFormer-подобная нормированная билинейная агрегация; полная архитектура статьи не воспроизводится.
    Затем Z = beta * Z + (1 - beta) * x  (residual с весом beta).
    """

    def __init__(self, d_model: int, heads: int = 1, beta: float = 0.5):
        super().__init__()
        if d_model % heads or not 0 <= beta <= 1:
            raise ValueError("d_model должен делиться на heads; beta должна быть в [0, 1]")
        self.heads, self.d_head = heads, d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        N = x.size(0)
        q = self.q(x).view(N, self.heads, self.d_head)
        k = self.k(x).view(N, self.heads, self.d_head)
        v = self.v(x).view(N, self.heads, self.d_head)
        q = q / (q.pow(2).sum(dim=(0, 2), keepdim=True).sqrt() + 1e-6)   # норма Фробениуса по голове
        k = k / (k.pow(2).sum(dim=(0, 2), keepdim=True).sqrt() + 1e-6)
        kv = torch.einsum("nhd,nhe->hde", k, v)                        # [H, d_head, d_head]
        num = v + torch.einsum("nhd,hde->nhe", q, kv) / N              # [N, H, d_head]
        k_sum = k.sum(dim=0)                                           # [H, d_head]
        den = 1.0 + torch.einsum("nhd,hd->nh", q, k_sum) / N           # [N, H]
        z = (num / den.unsqueeze(-1)).reshape(N, -1)
        return self.beta * z + (1.0 - self.beta) * x


# ---------------------------------------------------------------------------
# MPNN-слой (GINE) и гибридный слой GPS
# ---------------------------------------------------------------------------
class GINELayer(nn.Module):
    """GINE (GIN с признаками рёбер) + LayerNorm + residual: x + LN(GINE(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, use_edge_attr: bool = True):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.use_edge_attr = use_edge_attr
        self.conv = (GINEConv(mlp, train_eps=True, edge_dim=d_model) if use_edge_attr
                     else GINConv(mlp, train_eps=True))
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def delta(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None) -> Tensor:
        """Приращение без residual (нужно слою GPS, который складывает две ветви с одним residual)."""
        out = self.conv(x, edge_index, edge_attr) if self.use_edge_attr else self.conv(x, edge_index)
        return self.dropout(self.ln(F.relu(out)))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        return x + self.delta(x, edge_index, edge_attr)


class GPSLayer(nn.Module):
    """Слой GraphGPS (Rampášek et al., 2022): локальный MPNN и глобальное внимание параллельно.

    x' = x + Δ_MPNN(x) + Δ_Attn(x);  x'' = x' + FFN(LN(x')),
    где Δ_MPNN = Dropout(LN(ReLU(GINE(x, A, E)))), Δ_Attn = Dropout(Attn(LN(x))) — один общий residual
    на две ветви. Это GPS-подобная учебная архитектура, а не точная реплика GraphGPS.
    Локальная часть работает в разреженном (PyG) представлении, глобальная — в плотном;
    функции ``to_dense``/``from_dense`` передаёт модель (``models.GPS``).
    """

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float, attn_dropout: float,
                 use_edge_attr: bool = True):
        super().__init__()
        self.local = GINELayer(d_model, d_ff, dropout, use_edge_attr=use_edge_attr)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = DenseAttention(d_model, heads, attn_dropout)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                to_dense, from_dense, node_mask: Tensor) -> Tensor:
        d_local = self.local.delta(x, edge_index, edge_attr)
        h = to_dense(x)
        d_global = from_dense(self.dropout(self.attn(self.ln_attn(h), node_mask)))
        x = x + d_local + d_global
        x = x + self.dropout(self.ff(self.ln_ff(x)))
        return x

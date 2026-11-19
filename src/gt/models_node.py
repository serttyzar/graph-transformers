"""Модели для классификации узлов на большом графе (ogbn-arxiv, 169 тыс. узлов) — класс E таксономии.

Полное внимание по всем узлам здесь невозможно (N^2 ≈ 3·10^10 пар), поэтому реализованы
четыре способа обойти квадратичность:
- ``GCN``            — настроенный MPNN-бейзлайн (residual, LayerNorm, dropout; Luo et al., 2024);
- ``SGFormerLike``   — один слой линейного глобального внимания + ветка GCN (Wu et al., 2023);
- ``NAGphormerLike`` — узел -> последовательность агрегатов по хопам (Hop2Token), трансформер по хопам,
                       мини-батчи узлов (Chen et al., 2023);
- ``ExphormerLike``  — разреженное внимание по рёбрам графа ∪ рёбрам случайного экспандера ∪ рёбрам
                       к виртуальным узлам (Shirzad et al., 2023), параллельно с GCN-веткой.
Все модели принимают признаки узлов x [N, F] (вещественные) и edge_index.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, coalesce, remove_self_loops, to_undirected

from .layers import FeedForward, LinearAttention, SparseAttention, TransformerLayer


# ---------------------------------------------------------------------------
# Вспомогательные построения графов
# ---------------------------------------------------------------------------
def gcn_adjacency_cpu(edge_index: Tensor, num_nodes: int) -> Tensor:
    """CSR adj_t: GCN использует SpMM без буфера |E|×hidden.

    Убираем диагональ: gcn_norm для sparse добавит ровно одну петлю на узел.
    Это соответствует add_remaining_self_loops обычного GCNConv с весами 1.
    """
    edges, _ = remove_self_loops(edge_index.cpu())
    return torch.sparse_coo_tensor(edges.flip(0), torch.ones(edges.size(1)),
                                   (num_nodes, num_nodes)).coalesce().to_sparse_csr()


def expander_edges(num_nodes: int, degree: int, generator: torch.Generator | None = None) -> Tensor:
    """Случайный регулярный «экспандер» степени ``degree`` как объединение degree/2 случайных
    перестановок (каждая перестановка даёт цикл; сумма нескольких случайных перестановок
    с высокой вероятностью является экспандером — конструкция из Exphormer). Возвращает
    edge_index [2, num_nodes * degree] (рёбра в обе стороны).
    """
    rows, cols = [], []
    if degree < 0 or degree % 2:
        raise ValueError("Степень экспандера должна быть неотрицательной и чётной")
    if degree == 0:
        return torch.empty((2, 0), dtype=torch.long)
    idx = torch.arange(num_nodes)
    for _ in range(degree // 2):
        perm = torch.randperm(num_nodes, generator=generator)
        rows += [idx, perm]
        cols += [perm, idx]
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def virtual_node_edges(num_nodes: int, num_virtual: int) -> tuple[Tensor, int]:
    """Рёбра между всеми узлами и ``num_virtual`` виртуальными узлами (индексы N .. N+V-1) в обе стороны."""
    real = torch.arange(num_nodes).repeat(num_virtual)
    virt = torch.arange(num_nodes, num_nodes + num_virtual).repeat_interleave(num_nodes)
    ei = torch.stack([torch.cat([real, virt]), torch.cat([virt, real])])
    return ei, num_nodes + num_virtual


def build_interaction_graph(edge_index: Tensor, num_nodes: int, expander_degree: int,
                            num_virtual: int, seed: int = 0) -> tuple[Tensor, int]:
    """Граф взаимодействий H для Exphormer-подобного внимания: рёбра G ∪ экспандер ∪ виртуальные узлы ∪ петли."""
    g = torch.Generator().manual_seed(seed)
    parts = [to_undirected(edge_index.cpu(), num_nodes=num_nodes)]
    if expander_degree > 0:
        parts.append(expander_edges(num_nodes, expander_degree, g))
    total = num_nodes
    if num_virtual > 0:
        ve, total = virtual_node_edges(num_nodes, num_virtual)
        parts.append(ve)
    ei = torch.cat(parts, dim=1)
    ei, _ = add_remaining_self_loops(ei, num_nodes=total)
    return coalesce(ei, num_nodes=total), total


@torch.no_grad()
def hop_features(x: Tensor, edge_index: Tensor, num_hops: int) -> Tensor:
    """Hop2Token (NAGphormer): [N, K+1, F] — x, Âx, Â²x, ..., Â^K x, где Â = D^{-1/2}(A+I)D^{-1/2}.

    Считается один раз разреженными умножениями (O(K·|E|·F)); модель затем работает
    с узлами независимо, что и позволяет обучать мини-батчами без сэмплирования соседей.
    """
    N = x.size(0)
    ei, _ = add_remaining_self_loops(to_undirected(edge_index, num_nodes=N), num_nodes=N)
    deg = torch.zeros(N, device=x.device).index_add_(0, ei[1], torch.ones(ei.size(1), device=x.device))
    w = deg[ei[0]].pow(-0.5) * deg[ei[1]].pow(-0.5)
    A_hat = torch.sparse_coo_tensor(ei, w, (N, N)).coalesce()
    feats = [x]
    cur = x
    for _ in range(num_hops):
        cur = torch.sparse.mm(A_hat, cur)
        feats.append(cur)
    return torch.stack(feats, dim=1)


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Контроль «без структуры»: только признаки узлов, граф не используется."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 3, dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        h = x
        for lin, norm in zip(self.lins, self.norms):
            h = F.dropout(F.relu(norm(lin(h))), self.dropout, self.training)
        return self.out(h)


class GCN(nn.Module):
    """GCN с residual-связями, LayerNorm и dropout (рецепт «классические GNN — сильные бейзлайны»)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(GCNConv(hidden, hidden, cached=True) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.dropout(self.inp(x), self.dropout, self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, edge_index))), self.dropout, self.training)
        return self.out(h)


class SGFormerLike(nn.Module):
    """SGFormer: Z = LinearAttention(x W); Z_O = (1 - alpha) Z + alpha GCN(x); линейно по N."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, gcn_layers: int = 3,
                 dropout: float = 0.5, alpha: float = 0.5, beta: float = 0.5, heads: int = 1):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha должна быть в [0, 1]")
        self.inp = nn.Linear(in_dim, hidden)
        self.attn = LinearAttention(hidden, heads=heads, beta=beta)
        self.ln = nn.LayerNorm(hidden)
        # Общие блоки получают одинаковую инициализацию при одном seed в ±GCN контролях.
        self.out = nn.Linear(hidden, out_dim)
        self.gcn = GCN(in_dim, hidden, hidden, gcn_layers, dropout) if alpha > 0 else None
        self.alpha, self.dropout = alpha, dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        z = self.ln(self.attn(h))
        z = F.dropout(F.relu(z), self.dropout, self.training)
        if self.gcn is not None:
            z = (1 - self.alpha) * z + self.alpha * self.gcn(x, edge_index)
        return self.out(z)


class NAGphormerLike(nn.Module):
    """Трансформер по последовательности хопов узла с attention-based readout.

    Вход: [B, K+1, F] (см. ``hop_features``); выход — логиты классов узла.
    Readout: веса хопов a_k = softmax(w^T [h_0 || h_k]), результат h_0 + Σ_k a_k h_k.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_hops: int, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.hop_emb = nn.Parameter(torch.zeros(1, num_hops + 1, hidden))   # позиция хопа в последовательности
        self.layers = nn.ModuleList(TransformerLayer(hidden, heads, 2 * hidden, dropout, attn_dropout)
                                    for _ in range(layers))
        self.ln = nn.LayerNorm(hidden)
        self.readout_w = nn.Linear(2 * hidden, 1)
        self.out = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out_dim))
        nn.init.normal_(self.hop_emb, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        h = self.inp(tokens) + self.hop_emb                                    # [B, K+1, d]
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.ln(h)
        h0, hk = h[:, :1], h[:, 1:]                                           # [B,1,d], [B,K,d]
        a = torch.softmax(self.readout_w(torch.cat([h0.expand_as(hk), hk], -1)), dim=1)  # [B,K,1]
        g = h0.squeeze(1) + (a * hk).sum(1)
        return self.out(g)


class ExphormerLike(nn.Module):
    """Разреженное внимание по графу взаимодействий (рёбра ∪ экспандер ∪ виртуальные узлы)
    параллельно с GCN-веткой (как в GPS). Память O(|E_H|·H), |E_H| = O(N + |E|)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2, heads: int = 4,
                 dropout: float = 0.3, attn_dropout: float = 0.1, num_virtual: int = 4):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.virtual = nn.Parameter(torch.zeros(num_virtual, hidden))
        self.attn_layers = nn.ModuleList(SparseAttention(hidden, heads, attn_dropout) for _ in range(layers))
        self.attn_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.gcn_layers = nn.ModuleList(GCNConv(hidden, hidden, cached=True) for _ in range(layers))
        self.gcn_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ff_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ffs = nn.ModuleList(FeedForward(hidden, 2 * hidden, dropout) for _ in range(layers))
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.num_virtual = num_virtual
        nn.init.normal_(self.virtual, std=0.02)

    def forward(self, x: Tensor, edge_index: Tensor, interaction_index: Tensor) -> Tensor:
        N = x.size(0)
        h = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        h = torch.cat([h, self.virtual], dim=0)                                  # + виртуальные узлы
        for i in range(len(self.attn_layers)):
            d_attn = F.dropout(self.attn_layers[i](self.attn_norms[i](h), interaction_index),
                               self.dropout, self.training)
            d_gcn = torch.zeros_like(h)                                          # виртуальные узлы GCN не видят
            d_gcn[:N] = F.dropout(F.relu(self.gcn_norms[i](self.gcn_layers[i](h[:N], edge_index))),
                                  self.dropout, self.training)
            h = h + d_attn + d_gcn
            h = h + F.dropout(self.ffs[i](self.ff_norms[i](h)), self.dropout, self.training)
        return self.out(h[:N])

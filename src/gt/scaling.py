"""E4а: измерение forward+backward; каждая точка запускается отдельным процессом.

У hop-token один проход по N узлам состоит из мини-батчей. Хранилище токенов на CPU;
объём хранилища, время и пиковая GPU-память подготовки учитываются отдельно.
"""
from __future__ import annotations

import statistics

import torch
from torch import nn

from .data import synthetic_graph
from .layers import DenseAttention, FeedForward, LinearAttention, SparseAttention
from .models_node import NAGphormerLike, build_interaction_graph, hop_features
from .utils import Timer, host_peak_rss_mb, peak_gpu_mem_mb, reset_peak_mem, set_seed


class _Stack(nn.Module):
    def __init__(self, attn_layers, d_model):
        super().__init__()
        self.attn = nn.ModuleList(attn_layers)
        self.ln1 = nn.ModuleList(nn.LayerNorm(d_model) for _ in attn_layers)
        self.ln2 = nn.ModuleList(nn.LayerNorm(d_model) for _ in attn_layers)
        self.ff = nn.ModuleList(FeedForward(d_model, 2 * d_model, 0.0) for _ in attn_layers)

    def forward(self, x, attn_call):
        for a, l1, l2, ff in zip(self.attn, self.ln1, self.ln2, self.ff):
            x = x + attn_call(a, l1(x))
            x = x + ff(l2(x))
        return x


def build_method(method, data, d_model, heads, layers, device, hops=4,
                 expander_degree=6, num_virtual=4, hop_batch_size=2048, seed=0):
    """Возвращает (step, setup_info); каждый step обнуляет градиенты."""
    n = data.num_nodes
    meta = dict(precompute_s=0.0, token_storage_mb=0.0)
    if method == 'hop-token':
        with Timer() as timer:
            tokens = hop_features(data.x, data.edge_index, hops)
        meta.update(precompute_s=timer.elapsed,
                    token_storage_mb=tokens.numel() * tokens.element_size() / 2**20)
        model = NAGphormerLike(data.x.size(1), d_model, 1, hops, layers=layers, heads=heads,
                              dropout=0.0, attn_dropout=0.0).to(device)

        def step():
            model.zero_grad(set_to_none=True)
            for chunk in tokens.split(hop_batch_size):
                (model(chunk.to(device)).sum() / n).backward()

        return step, meta

    x = data.x.to(device)
    inp = nn.Linear(x.size(1), d_model).to(device)
    if method == 'full':
        stack = _Stack([DenseAttention(d_model, heads) for _ in range(layers)], d_model).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)

        def forward():
            return stack(inp(x).unsqueeze(0), lambda a, t: a(t, mask))

    elif method in ('sparse-neigh', 'sparse-expander'):
        with Timer() as timer:
            degree, virtual = (0, 0) if method == 'sparse-neigh' else (expander_degree, num_virtual)
            ei, total = build_interaction_graph(data.edge_index, n, degree, virtual, seed=seed)
            ei = ei.to(device)
        meta['precompute_s'] = timer.elapsed
        stack = _Stack([SparseAttention(d_model, heads) for _ in range(layers)], d_model).to(device)
        extra = torch.zeros(total - n, d_model, device=device)

        def forward():
            return stack(torch.cat([inp(x), extra]), lambda a, t: a(t, ei))

    elif method == 'linear':
        stack = _Stack([LinearAttention(d_model, heads=heads) for _ in range(layers)], d_model).to(device)

        def forward():
            return stack(inp(x), lambda a, t: a(t) - t)

    else:
        raise ValueError(method)

    def step():
        inp.zero_grad(set_to_none=True)
        stack.zero_grad(set_to_none=True)
        (forward().sum() / n).backward()

    return step, meta


def benchmark(method, num_nodes, device, d_model=64, heads=4, layers=2, avg_degree=10,
              repeats=5, warmup=2, seed=0, hop_batch_size=2048):
    set_seed(seed)
    reset_peak_mem()
    data = synthetic_graph(num_nodes, avg_degree, feat_dim=d_model, seed=seed)
    rec = dict(method=method, n_nodes=num_nodes, n_edges=int(data.edge_index.size(1)),
               d_model=d_model, heads=heads, layers=layers, hop_batch_size=hop_batch_size, status='ok')
    step = None
    try:
        step, setup = build_method(method, data, d_model, heads, layers, device,
                                  hop_batch_size=hop_batch_size, seed=seed)
        rec.update(setup)
        rec['setup_peak_mem_mb'] = peak_gpu_mem_mb()
        for _ in range(warmup):
            step()
        reset_peak_mem()
        times = []
        for _ in range(repeats):
            with Timer() as timer:
                step()
            times.append(timer.elapsed)
        rec.update(fwd_bwd_time_s=statistics.median(times), peak_mem_mb=peak_gpu_mem_mb())
    except torch.cuda.OutOfMemoryError:
        rec['status'] = 'oom'
    finally:
        rec['host_peak_rss_mb'] = host_peak_rss_mb()
        del step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec

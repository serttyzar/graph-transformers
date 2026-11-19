"""Расчёты сводных таблиц основной серии. Функции принимают записи запусков."""
import json
import statistics
from collections import Counter

import numpy as np
from scipy import stats


def by_seed(runs, field='test_metric_at_best_val'):
    """Одна оценка на seed; фолды исторических fixtures усредняются внутри seed."""
    return {seed: float(np.mean([r[field] for r in runs if r['seed'] == seed]))
            for seed in sorted({r['seed'] for r in runs})}


def paired_interval(left, right):
    if set(left) != set(right) or len(left) < 2:
        raise ValueError('Нужны >=2 совпадающих seeds')
    delta = np.array([left[s] - right[s] for s in sorted(left)])
    mean = float(delta.mean())
    margin = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
    return mean, mean - margin, mean + margin, len(delta)


def summarize(cfg, runs, variants):
    rows = []
    for variant in variants:
        group = [r for r in runs if r['variant'] == variant]
        scores = np.array(list(by_seed(group).values()))
        val = np.array(list(by_seed(group, 'best_val_metric').values()))
        widths = sorted({r['config']['model'].get('d_model', r['config']['model'].get('hidden')) for r in group})
        rows.append(dict(experiment=cfg['experiment'], variant=variant, n_runs=len(group),
            n_seeds=len(scores), folds=cfg.get('folds', 1), test_mean=float(scores.mean()),
            test_std=float(scores.std(ddof=1)), val_mean=float(val.mean()),
            n_params_min=min(r['n_params'] for r in group), n_params_max=max(r['n_params'] for r in group),
            widths='/'.join(map(str, widths)),
            time_per_epoch_s_mean=float(np.mean([r['time_per_epoch_s'] for r in group])),
            train_time_total_s_mean=float(np.mean([r['train_time_total_s'] for r in group])),
            search_time_total_s_mean=float(np.mean([r.get('search_time_total_s', r['train_time_total_s']) for r in group])),
            precompute_s_mean=float(np.mean([r.get('precompute_s', 0) for r in group])),
            peak_gpu_mem_mb_max=max(r['peak_gpu_mem_mb'] for r in group),
            host_peak_rss_mb_max=max(r.get('host_peak_rss_mb', 0) for r in group)))
    return rows


def scaling_slopes(runs):
    rows = []
    for method in sorted({r['method'] for r in runs}):
        ok = [r for r in runs if r['method'] == method and r['status'] == 'ok']
        sizes = sorted({r['n_nodes'] for r in ok})
        memory_key = 'host_peak_rss_mb' if ok and all(r.get('gpu_name') == 'cpu' for r in ok) else 'peak_mem_mb'
        for key in ['fwd_bwd_time_s', memory_key]:
            points = [(n, float(np.median([r.get(key, 0) for r in ok if r['n_nodes'] == n]))) for n in sizes]
            points = [(n, value) for n, value in points if value > 0]
            if len(points) < 3:
                continue
            n, y = np.array(points).T
            fit = stats.linregress(np.log(n), np.log(y))
            rows.append(dict(hypothesis='H4', method=method, metric=key, alpha=float(fit.slope),
                alpha_stderr=float(fit.stderr), n_points=len(points), min_n=int(n.min()), max_n=int(n.max()),
                note='finite range fit; constant width/hops/average degree; failed/guarded points excluded; RSS includes setup'))
    return rows


def median_points(runs):
    """Одна строка на метод/размер; guarded значения остаются пустыми."""
    rows = []
    for method in sorted({r['method'] for r in runs}):
        for n in sorted({r['n_nodes'] for r in runs}):
            group = [r for r in runs if r['method'] == method and r['n_nodes'] == n]
            ok = [r for r in group if r['status'] == 'ok']
            row = dict(method=method, n_nodes=n, n_ok=len(ok),
                       statuses=json.dumps(dict(Counter(r['status'] for r in group))))
            for key in ('fwd_bwd_time_s', 'peak_mem_mb', 'precompute_s',
                        'setup_peak_mem_mb', 'token_storage_mb', 'host_peak_rss_mb'):
                row[key] = statistics.median(r.get(key, 0) for r in ok) if ok else None
            rows.append(row)
    return rows


def common_slopes(runs):
    methods = sorted({r['method'] for r in runs})
    shared = set.intersection(*[{r['n_nodes'] for r in runs
                               if r['method'] == method and r['status'] == 'ok'}
                              for method in methods])
    if len(shared) < 3:
        return []
    return scaling_slopes([r for r in runs if r['n_nodes'] in shared])

"""Один прогон: YAML + вариант + seed -> проверенный атомарный JSON.

--smoke использует настоящий датасет; --offline-smoke — синтетические данные без сети.
--force сохраняет прежний JSON в _superseded; устаревшие результаты не пропускаются молча.
"""
from __future__ import annotations

import argparse
import copy
import os
import socket
import sys
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gt import data as gdata
from gt import models as gmodels
from gt import models_node as nmodels
from gt import smoke_data
from gt.train import evaluate_node, train_graph_level, train_node_level
from gt.utils import (Timer, archive_result, code_version, config_hash, env_info,
                      get_device, host_peak_rss_mb, load_json, peak_gpu_mem_mb, save_json, set_seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--results-root', default=str(ROOT / 'results' / 'compact_v3' / 'raw'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--offline-smoke', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true', help='разрешить полные прогоны без CUDA')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--tag', default='')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--validation-only', action='store_true', help='служебная проверка без доступа к test')
    return parser.parse_args()


def run_graph_experiment(cfg, args):
    mcfg = dict(copy.deepcopy(cfg['base']), **cfg['variants'][args.variant])
    tcfg = copy.deepcopy(cfg['train'])
    if args.smoke:
        tcfg.update(epochs=2, log_every=1)
    if args.epochs is not None:
        tcfg.update(epochs=args.epochs, log_every=1)
    lappe_k = mcfg['lappe_k'] if 'lappe' in (mcfg.get('node_enc') or []) else 0
    rwse_k = mcfg['rwse_k'] if 'rwse' in (mcfg.get('node_enc') or []) else 0
    with Timer() as setup:
        if args.offline_smoke:
            splits, info = smoke_data.graph_splits(cfg['dataset'], lappe_k, rwse_k)
        else:
            splits, info = gdata.load_graph_dataset(cfg['dataset'], args.data_root, lappe_k, rwse_k,
                                                     max_graphs=3 * tcfg['batch_size'] if args.smoke else None)
        if cfg['dataset'] == 'csl':
            if args.fold is None:
                raise ValueError('Для CSL укажите --fold')
            splits = gdata.csl_folds(splits['all'], args.fold, cfg.get('folds', 5))
    set_seed(args.seed)
    mcfg = gmodels.fit_budget(mcfg, info, cfg['budget'])
    set_seed(args.seed)  # fit_budget создаёт временные модели и расходует RNG
    model = gmodels.build_model(mcfg, info)
    count = gmodels.count_params(model)
    print(f"[{cfg['experiment']}/{args.variant}] seed={args.seed} params={count} "
          f"d_model={mcfg['d_model']} d_ff={mcfg['d_ff']}", flush=True)
    res = train_graph_level(model, splits, info, tcfg, get_device(),
                            evaluate_test=not args.validation_only)
    persist_best(model, cfg, args, mcfg, info)
    return dict(config=dict(model=mcfg, train=tcfg), n_params=count,
                data_setup_time_s=setup.elapsed, **res)


def run_arxiv_experiment(cfg, args):
    vcfg = copy.deepcopy(cfg['variants'][args.variant])
    tcfg = copy.deepcopy(cfg['train'])
    for key in ('lr', 'epochs', 'patience', 'weight_decay', 'eval_batch_size', 'scheduler', 'warmup_frac', 'min_epochs'):
        if key in vcfg:
            tcfg[key] = vcfg.pop(key)
    candidates = vcfg.pop('lr_candidates', [tcfg['lr']])
    if args.smoke:
        tcfg.update(epochs=2, log_every=1)
    if args.epochs is not None:
        tcfg.update(epochs=args.epochs, log_every=1)
    device = get_device()
    kind = vcfg.pop('model')
    with Timer() as setup:
        data, split, info = (smoke_data.node_data() if args.offline_smoke
                             else gdata.load_ogbn_arxiv(args.data_root))
        x = data.x if kind == 'nagphormer' else data.x.to(device)
        y = data.y.to(device)
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        if kind in ('gcn', 'exphormer') or (kind == 'sgformer' and vcfg['alpha'] > 0):
            edge_index = edge_index.to(device)
        split = {k: v.to(device) for k, v in split.items()}
    tokens, precompute_s = None, 0.0
    minibatch = (vcfg['batch_size'],) if kind == 'nagphormer' else None
    if minibatch:
        with Timer() as pre:
            if vcfg.get('structure', True):
                tokens = nmodels.hop_features(data.x, edge_index.cpu(), vcfg['hops'])
            else:
                # Те же K+1 токенов и та же архитектура; A не используется.
                tokens = data.x[:, None, :].repeat(1, vcfg['hops'] + 1, 1)
        precompute_s = pre.elapsed
    inter = None
    if kind == 'exphormer':
        with Timer() as pre:
            inter, _ = nmodels.build_interaction_graph(data.edge_index, data.num_nodes,
                vcfg['expander_degree'], vcfg['num_virtual'], seed=args.seed)
            inter = inter.to(device)
        precompute_s = pre.elapsed
    if device.type == 'cpu' and (kind in ('gcn', 'exphormer') or
                               (kind == 'sgformer' and vcfg['alpha'] > 0)):
        edge_index = nmodels.gcn_adjacency_cpu(edge_index, data.num_nodes)

    def build():
        if kind == 'mlp':
            return nmodels.MLP(info['node_dim'], vcfg['hidden'], info['out_dim'], vcfg['layers'], vcfg['dropout'])
        if kind == 'gcn':
            return nmodels.GCN(info['node_dim'], vcfg['hidden'], info['out_dim'], vcfg['layers'], vcfg['dropout'])
        if kind == 'sgformer':
            return nmodels.SGFormerLike(info['node_dim'], vcfg['hidden'], info['out_dim'],
                vcfg['gcn_layers'], vcfg['dropout'], vcfg['alpha'], vcfg['beta'])
        if kind == 'nagphormer':
            return nmodels.NAGphormerLike(info['node_dim'], vcfg['hidden'], info['out_dim'],
                vcfg['hops'], vcfg['layers'], vcfg['heads'], vcfg['dropout'], vcfg['attn_dropout'])
        if kind == 'exphormer':
            return nmodels.ExphormerLike(info['node_dim'], vcfg['hidden'], info['out_dim'],
                vcfg['layers'], vcfg['heads'], vcfg['dropout'], vcfg['attn_dropout'], vcfg['num_virtual'])
        raise ValueError(kind)

    def forward(idx):
        if kind == 'mlp':
            return model(x)
        if kind == 'nagphormer':
            # Хранилище токенов на CPU; GPU получает только текущий батч.
            return model(tokens[idx.cpu()].to(device))
        if kind == 'exphormer':
            return model(x, edge_index, inter)
        return model(x, edge_index)

    # Для каждого seed кандидаты стартуют с одной инициализации. Test не вызывается при поиске.
    selected, state, trials = None, None, []
    search_time, search_peak = 0.0, 0.0
    for lr in candidates:
        set_seed(args.seed)
        model = build().to(device)
        trial_cfg = dict(tcfg, lr=lr)
        result = train_node_level(model, forward, y, split, trial_cfg, minibatch=minibatch, evaluate_test=False)
        search_time += result['train_time_total_s']
        search_peak = max(search_peak, result['peak_gpu_mem_mb'])
        trials.append(dict(lr=lr, best_val_metric=result['best_val_metric'], best_epoch=result['best_epoch']))
        if selected is None or result['best_val_metric'] > selected['best_val_metric']:
            selected = dict(result, selected_lr=lr)
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model
    model = build().to(device)
    model.load_state_dict(state)
    selected['test_metric_at_best_val'] = (evaluate_node(model, forward, y, split['test'],
        tcfg.get('eval_batch_size', minibatch[0]) if minibatch else None) if not args.validation_only else None)
    selected['test_evaluations'] = int(not args.validation_only)
    selected['peak_gpu_mem_mb'] = max(search_peak, peak_gpu_mem_mb())
    persist_best(model, cfg, args, vcfg, info)
    tcfg['lr'] = selected.pop('selected_lr')
    return dict(config=dict(model=dict(kind=kind, **vcfg), train=tcfg),
                n_params=gmodels.count_params(model), data_setup_time_s=setup.elapsed,
                node_adjacency_layout=str(edge_index.layout),
                data_provenance=info.get('data_provenance'),
                precompute_s=precompute_s, search_time_total_s=search_time, lr_search=trials,
                token_storage_mb=(tokens.numel() * tokens.element_size() / 2**20 if tokens is not None else 0),
                **selected)


def persist_best(model, cfg, args, model_config, info):
    # Seed 0 каждой модели — достаточно для демонстрации/повторного inference;
    # остальные seed воспроизводятся по коду и полному конфигу. Не сохраняем pilot.
    if cfg['protocol_version'] != '4.0' or args.seed != 0 or args.validation_only or args.epochs is not None or args.smoke:
        return
    path = Path(args.results_root).parent / 'checkpoints' / cfg['experiment'] / f'{args.variant}_seed0.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    model_config=model_config, info=info, source_config=cfg, seed=args.seed,
                    code_version=code_version()), tmp)
    tmp.replace(path)


def main():
    socket.setdefaulttimeout(45)  # скачивание не должно зависать бесконечно
    args = parse_args()
    args.smoke = args.smoke or args.offline_smoke
    if args.smoke:
        torch.set_num_threads(1)
    if os.environ.get('COURSEWORK_THREADS'):
        torch.set_num_threads(int(os.environ['COURSEWORK_THREADS']))
    if not args.smoke and not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError('Для основной серии нужен CUDA GPU. Для отладки: --offline-smoke или --allow-cpu')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('--epochs >= 1')
    if any(c in args.tag for c in ('/', '\\')):
        raise ValueError('tag не должен содержать разделители пути')
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg.get('kind', 'training') != 'training':
        raise ValueError('Этот конфиг не обучающий; используйте run_diagnostic.py или run_scaling.py')
    if args.seed not in cfg['seeds']:
        raise ValueError('seed должен входить в текущий YAML')
    if args.variant not in cfg['variants']:
        raise ValueError(f"Нет варианта {args.variant}; есть {list(cfg['variants'])}")
    if cfg['dataset'] != 'csl' and args.fold is not None:
        raise ValueError('--fold допустим только для CSL')
    purpose = ('offline_smoke' if args.offline_smoke else 'smoke' if args.smoke else
               'calibration' if args.validation_only or args.epochs is not None or args.tag.startswith('calib') else 'research')
    request = dict(source_config=cfg, epochs_override=args.epochs, purpose=purpose)
    signature = config_hash(request)
    suffix = '_offline_smoke' if args.offline_smoke else '_smoke' if args.smoke else ''
    name = f'{args.variant}_seed{args.seed}' + (f'_fold{args.fold}' if args.fold is not None else '')
    name += suffix + ('_' + args.tag if args.tag else '')
    if args.epochs is not None and not args.tag:
        name += f'_calib_epochs{args.epochs}'
    out_path = str(Path(args.results_root) / cfg['experiment'] / (name + '.json'))
    if os.path.exists(out_path) and not args.force:
        existing = load_json(out_path)
        if (existing.get('status') == 'ok' and existing.get('config_hash') == signature
                and existing.get('code_version') == code_version()
                and existing.get('gpu_name') == env_info()['gpu_name']):
            print(f'уже есть проверенный результат: {out_path}')
            return
        raise RuntimeError(f'Результат устарел или неполон: {out_path}. Для пересчёта с архивированием: --force')
    start = time.perf_counter()
    res = run_arxiv_experiment(cfg, args) if cfg['dataset'] == 'ogbn-arxiv' else run_graph_experiment(cfg, args)
    record = dict(run_id=name, status='ok', experiment=cfg['experiment'], dataset=cfg['dataset'],
                  variant=args.variant, seed=args.seed, fold=args.fold, purpose=purpose,
                  protocol_version=cfg['protocol_version'], source_config=cfg, config_hash=signature,
                  smoke=args.smoke, elapsed_total_s=time.perf_counter() - start,
                  host_peak_rss_mb=host_peak_rss_mb(), **res, **env_info())
    if args.force:
        archive_result(out_path)
    save_json(record, out_path)
    print(f'сохранено: {out_path}', flush=True)


if __name__ == '__main__':
    main()

"""E4а: отдельный subprocess для каждой точки, включая повторы после OOM."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gt.utils import archive_result, code_version, config_hash, env_info, get_device, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'E4_scaling.yaml'))
    parser.add_argument('--results-root', default=str(ROOT / 'results' / 'compact_v3' / 'raw'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--method', help=argparse.SUPPRESS)
    parser.add_argument('--nodes', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--rep', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    signature = config_hash(dict(source_config=cfg, smoke=args.smoke))
    out_dir = Path(args.results_root) / (cfg['experiment'] + ('_smoke' if args.smoke else ''))

    def path_for(method, n, rep):
        return out_dir / f'{method}_n{n}_rep{rep}.json'

    def save(rec, method, n, rep, source_env=None):
        path = path_for(method, n, rep)
        metadata = env_info() if source_env is None else {k: source_env[k] for k in (
            'python_version', 'torch_version', 'pyg_version', 'numpy_version', 'scipy_version',
            'sklearn_version', 'cuda_version', 'gpu_name', 'code_version')}
        metadata['timestamp'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        rec.update(method=method, n_nodes=n, rep=rep, experiment=cfg['experiment'],
                   source_config=cfg, config_hash=signature, smoke=args.smoke,
                   purpose='smoke' if args.smoke else 'research', protocol_version=cfg['protocol_version'],
                   **metadata)
        if args.force:
            archive_result(str(path))
        save_json(rec, str(path))
        print(f"{method:16s} n={n:8d} rep={rep} -> {rec['status']}", flush=True)

    if args.worker:
        import torch
        from gt.scaling import benchmark
        if not args.smoke and not args.allow_cpu and not torch.cuda.is_available():
            raise RuntimeError('Для E4а нужен CUDA GPU; для отладки --smoke или --allow-cpu')
        if args.smoke:
            torch.set_num_threads(1)
        if os.environ.get('COURSEWORK_THREADS'):
            torch.set_num_threads(int(os.environ['COURSEWORK_THREADS']))
        if os.environ.get('COURSEWORK_HOST_LIMIT_GB'):
            import psutil
            from gt.local_resources import estimated_scaling_peak_mb, separate_scaling_estimates_mb
            if cfg.get('resource_guard') == 'separate_host_gpu':
                estimates = separate_scaling_estimates_mb(args.method, args.nodes, cfg['d_model'],
                    cfg['heads'], cfg['layers'], cfg['avg_degree'])
                limits = dict(host=min(float(os.environ['COURSEWORK_HOST_LIMIT_GB']) * 1024,
                    max(0, psutil.virtual_memory().available / 2**20 - 3072)))
                if torch.cuda.is_available():
                    limits['gpu'] = torch.cuda.mem_get_info()[0] / 2**20 * .85
                else:
                    estimates['host'] = max(estimates['host'], estimates['gpu'])
                for resource, limit in limits.items():
                    if estimates[resource] > limit:
                        save(dict(status='resource_limit', resource=resource,
                            estimated_peak_mb=estimates[resource], resource_limit_mb=limit,
                            estimates_mb=estimates, limits_mb=limits,
                            reason='Conservative pre-allocation guard; not observed OOM'),
                            args.method, args.nodes, args.rep)
                        return
            else:
                estimates = None
            estimate = estimated_scaling_peak_mb(args.method, args.nodes, cfg['d_model'],
                cfg['heads'], cfg['layers'], cfg['avg_degree'])
            limit = min(float(os.environ['COURSEWORK_HOST_LIMIT_GB']) * 1024,
                        max(0, psutil.virtual_memory().available / 2**20 - 3072))
            if cfg.get('resource_guard') != 'separate_host_gpu' and estimate > limit:
                save(dict(status='resource_limit', estimated_peak_mb=estimate,
                          host_memory_limit_mb=limit,
                          reason='Консервативная оценка превышает доступный бюджет RAM; вычисление не запускалось'),
                     args.method, args.nodes, args.rep)
                return
        rec = benchmark(args.method, args.nodes, get_device(), cfg['d_model'], cfg['heads'],
                        cfg['layers'], cfg['avg_degree'], cfg['repeats'], cfg['warmup'], seed=args.rep,
                        hop_batch_size=cfg['hop_batch_size'])
        save(rec, args.method, args.nodes, args.rep)
        return
    sizes = [64, 128] if args.smoke else cfg['num_nodes']
    reps = cfg['reps'][:1] if args.smoke else cfg['reps']
    for method in cfg['methods']:
        prior_oom = False
        oom_env = None
        for n in sizes:
            size_oom = False
            for rep in reps:
                path = path_for(method, n, rep)
                if path.exists() and not args.force:
                    rec = load_json(str(path))
                    if rec.get('config_hash') != signature or rec.get('code_version') != code_version():
                        raise RuntimeError(f'Устаревшая точка {path}; используйте --force')
                    if rec.get('status') not in ('ok', 'oom', 'skipped_after_oom', 'resource_limit'):
                        raise RuntimeError(f'Повреждённая точка {path}; используйте --force')
                    size_oom |= rec['status'] in ('oom', 'skipped_after_oom')
                    if rec['status'] == 'oom':
                        oom_env = rec
                    continue
                if prior_oom and cfg.get('skip_after_oom', True):
                    save(dict(status='skipped_after_oom'), method, n, rep, source_env=oom_env)
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--config', args.config,
                           '--results-root', args.results_root, '--method', method, '--nodes', str(n), '--rep', str(rep)]
                for flag in ('smoke', 'allow_cpu', 'force'):
                    if getattr(args, flag):
                        command.append('--' + flag.replace('_', '-'))
                subprocess.run(command, check=True)
                rec = load_json(str(path))
                size_oom |= rec['status'] == 'oom'
                if rec['status'] == 'oom':
                    oom_env = rec
            prior_oom |= size_oom


if __name__ == '__main__':
    main()

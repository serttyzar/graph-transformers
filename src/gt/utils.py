"""Сиды, устройство, таймеры, подсчёт параметров, версия кода, запись результатов."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    import torch
    requested = os.environ.get('COURSEWORK_DEVICE', 'auto')
    if requested not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('COURSEWORK_DEVICE: auto, cpu, mps или cuda')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS недоступен в этом процессе')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна в этом процессе')
    return torch.device(requested)


def code_version() -> str:
    """Хэш пакета и запускающих скриптов; YAML фиксируется отдельно config_hash."""
    root = Path(PACKAGE_DIR).parent
    h = hashlib.sha256()
    for folder in (root / "gt", root / "scripts"):
        for path in sorted(folder.iterdir()):
            if path.suffix in (".py", ".sh"):
                h.update(str(path.relative_to(root)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()[:16]


def config_hash(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def env_info() -> dict:
    import torch
    import platform
    import torch_geometric
    import scipy
    import sklearn
    return dict(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        pyg_version=torch_geometric.__version__,
        numpy_version=np.__version__, scipy_version=scipy.__version__, sklearn_version=sklearn.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if get_device().type == 'cuda' else str(get_device()),
        device=str(get_device()), machine=platform.machine(), platform=platform.platform(),
        torch_threads=torch.get_num_threads(),
        gpu_memory_metric='cuda_max_memory_allocated' if get_device().type == 'cuda' else 'unavailable',
        code_version=code_version(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def peak_gpu_mem_mb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**20


def reset_peak_mem() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class Timer:
    def __enter__(self):
        import torch
        synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import torch
        synchronize()
        self.elapsed = time.perf_counter() - self.t0


def synchronize():
    import torch
    device = get_device().type
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def host_peak_rss_mb():
    """Пиковый RSS всего процесса, включая данные и подготовку; не CUDA-память."""
    import resource
    import sys
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (2**20 if sys.platform == 'darwin' else 1024)


def save_json(obj: dict, path: str) -> None:
    """Атомарная запись: обрыв Colab не оставляет повреждённый файл готового прогона."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def archive_result(path: str) -> None:
    """Сохраняет прошлый результат при явном --force, не редактируя его содержимое."""
    target = Path(path)
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = target.parent / "_superseded" / f"{target.stem}_{stamp}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.replace(destination)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

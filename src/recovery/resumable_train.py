"""Checkpoint-enabled execution adapter for frozen gt/train.py (53cdb67277e393cb).

Base scientific sources remain immutable; this file has a separate runner hash.
Обучение с выбором весов по validation и однократной оценкой test.

Время шага оптимизации и полное время обучения учитываются отдельно.
Для независимых узловых примеров (Hop2Token) evaluation тоже выполняется батчами.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gt.utils import Timer, peak_gpu_mem_mb, reset_peak_mem, synchronize

HIGHER_IS_BETTER = {"mae": False, "acc": True, "ap": True}


def loss_fn(task: str, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if task == "regression":
        return F.l1_loss(out.view(-1), y.view(-1).float())
    if task == "classification":
        return F.cross_entropy(out, y.view(-1).long())
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(out, y.float().reshape_as(out))
    raise ValueError(task)


def compute_metric(metric: str, out: np.ndarray, y: np.ndarray) -> float:
    if not np.isfinite(out).all() or not np.isfinite(y).all():
        raise FloatingPointError("Метрика получила NaN/Inf")
    if metric == "mae":
        return float(np.mean(np.abs(out.reshape(-1) - y.reshape(-1))))
    if metric == "acc":
        return float(np.mean(out.argmax(axis=1) == y.reshape(-1)))
    if metric == "ap":
        from sklearn.metrics import average_precision_score
        y = y.reshape(out.shape)
        aps = [average_precision_score(y[:, j], out[:, j]) for j in range(y.shape[1])
               if y[:, j].sum() > 0 and (1 - y[:, j]).sum() > 0]
        if not aps:
            raise ValueError("AP не определена: нет метки с обоими классами")
        return float(np.mean(aps))
    raise ValueError(metric)


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    warmup = max(int(total_steps * warmup_frac), 1)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _step(loss, model, optimizer, grad_clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("Loss = NaN/Inf; результат прогона не будет сохранён")
    loss.backward()
    # error_if_nonfinite проверяет также переполнение градиентов при конечном loss.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or math.inf, error_if_nonfinite=True)
    optimizer.step()


@torch.no_grad()
def evaluate_graph(model, loader, info: dict, device) -> float:
    model.eval()
    outs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        outs.append(model(batch).float().cpu().numpy())
        ys.append(batch.y.float().cpu().numpy())
    return compute_metric(info["metric"], np.concatenate(outs), np.concatenate(ys))


def train_graph_level(model, splits: dict, info: dict, tcfg: dict, device, log=print,
                      evaluate_test=True, checkpoint=None) -> dict:
    model = model.to(device)
    if any(not splits.get(k) for k in ("train", "val", "test")) or tcfg["epochs"] < 1:
        raise ValueError("Нужны непустые train/val/test и epochs >= 1")
    train_loader = DataLoader(splits["train"], batch_size=tcfg["batch_size"], shuffle=True)
    eval_batch = tcfg.get("eval_batch_size", tcfg["batch_size"])
    val_loader = DataLoader(splits["val"], batch_size=eval_batch)
    test_loader = DataLoader(splits["test"], batch_size=eval_batch)
    train_eval_loader = DataLoader(splits["train"], batch_size=eval_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    sched = make_scheduler(opt, tcfg["epochs"] * len(train_loader), tcfg.get("warmup_frac", 0.05))
    higher = HIGHER_IS_BETTER[info["metric"]]
    best = dict(val=-math.inf if higher else math.inf, epoch=-1)
    history, epoch_times, epoch_wall_times, train_history, lr_history = [], [], [], [], []
    best_state = None
    reset_peak_mem()
    start = time.perf_counter()
    resume = checkpoint.restore(model, opt, sched, 'graph', tcfg) if checkpoint else None
    if resume:
        best, best_state = resume['best'], resume['best_state']
        history, epoch_times, epoch_wall_times = resume['history'], resume['epoch_times'], resume['epoch_wall_times']
        train_history, lr_history = resume['train_history'], resume['lr_history']
    elapsed_before = resume['elapsed_s'] if resume else 0.0
    previous_peak = resume['peak_mb'] if resume else 0.0
    for epoch in range(len(history) + 1, tcfg["epochs"] + 1):
        synchronize()
        epoch_started = time.perf_counter()
        model.train()
        total_loss, n_graphs = 0.0, 0
        with Timer() as timer:
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(info["task"], model(batch), batch.y)
                _step(loss, model, opt, tcfg.get("grad_clip"))
                sched.step()
                total_loss += loss.item() * batch.num_graphs
                n_graphs += batch.num_graphs
        epoch_times.append(timer.elapsed)
        train_history.append(total_loss / n_graphs)
        lr_history.append(opt.param_groups[0]['lr'])
        val = evaluate_graph(model, val_loader, info, device)
        history.append(val)
        improved = val > best["val"] if higher else val < best["val"]
        if improved:
            best.update(val=val, epoch=epoch)
            best_state = _cpu_state(model)
        synchronize()
        epoch_wall_times.append(time.perf_counter() - epoch_started)
        if epoch % tcfg.get("log_every", 10) == 0 or epoch == 1:
            log(f"эпоха {epoch:4d} | loss {total_loss / n_graphs:.4f} | val {info['metric']} {val:.4f} "
                f"| best val {best['val']:.4f} (эпоха {best['epoch']}) | {timer.elapsed:.2f} с")
        if checkpoint:
            checkpoint.save(model, opt, sched, 'graph', tcfg, dict(epoch=epoch,
                best=best, best_state=best_state, history=history, epoch_times=epoch_times,
                epoch_wall_times=epoch_wall_times, train_history=train_history, lr_history=lr_history,
                elapsed_s=elapsed_before + time.perf_counter() - start,
                peak_mb=max(previous_peak, peak_gpu_mem_mb())), final=epoch == tcfg["epochs"])
    if best_state is None:
        raise RuntimeError("Нет корректного validation checkpoint")
    model.load_state_dict(best_state)
    test = evaluate_graph(model, test_loader, info, device) if evaluate_test else None
    train = evaluate_graph(model, train_eval_loader, info, device)
    return dict(epochs_run=len(history), best_epoch=best["epoch"], best_val_metric=best["val"],
                test_metric_at_best_val=test, train_metric_at_best_val=train, test_evaluations=int(evaluate_test),
                train_loss_history=train_history, learning_rate_history=lr_history,
                history=history, train_time_total_s=elapsed_before + time.perf_counter() - start,
                optimization_time_total_s=float(sum(epoch_times)),
                epoch_wall_times_s=epoch_wall_times,
                time_per_epoch_s=float(np.mean(epoch_times)), peak_gpu_mem_mb=max(previous_peak, peak_gpu_mem_mb()))


@torch.no_grad()
def evaluate_node(model, forward_fn, y, idx, batch_size: int | None = None) -> float:
    """Полный граф либо батчи независимых узлов; logits всего сплита на GPU не хранятся."""
    model.eval()
    if idx.numel() == 0:
        raise ValueError("Пустой сплит")
    if batch_size is None:
        out = forward_fn(None)[idx]
        if not torch.isfinite(out).all():
            raise FloatingPointError("Logits = NaN/Inf")
        return float((out.argmax(1) == y[idx]).float().mean())
    correct = 0
    for chunk in idx.split(batch_size):
        out = forward_fn(chunk)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Logits = NaN/Inf")
        correct += int((out.argmax(1) == y[chunk]).sum())
    return correct / idx.numel()


def train_node_level(model, forward_fn, y: torch.Tensor, split: dict, tcfg: dict, log=print,
                     minibatch: tuple | None = None, evaluate_test: bool = True, checkpoint=None) -> dict:
    """Оставляет в model лучшие по validation веса.

    evaluate_test=False используется только для кандидатов lr-search: test не вызывается.
    После выбора кандидата вызывающий код оценивает test один раз.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    train_idx, val_idx = split["train"], split["valid"]
    if not train_idx.numel() or not val_idx.numel() or tcfg["epochs"] < 1:
        raise ValueError("Нужны непустые train/valid и epochs >= 1")
    eval_batch = tcfg.get("eval_batch_size", minibatch[0]) if minibatch else None
    best = dict(val=-1.0, epoch=-1)
    history, epoch_times, epoch_wall_times, bad, best_state = [], [], [], 0, None
    train_history, lr_history = [], []
    sched = (make_scheduler(opt, tcfg['epochs'] * (math.ceil(len(train_idx) / minibatch[0]) if minibatch else 1),
                            tcfg.get('warmup_frac', 0.05)) if tcfg.get('scheduler') == 'cosine' else None)
    reset_peak_mem()
    start = time.perf_counter()
    resume = checkpoint.restore(model, opt, sched, 'node', tcfg) if checkpoint else None
    if resume:
        best, best_state = resume['best'], resume['best_state']
        history, epoch_times, epoch_wall_times = resume['history'], resume['epoch_times'], resume['epoch_wall_times']
        train_history, lr_history = resume['train_history'], resume['lr_history']
    if resume:
        bad = resume['bad']
    elapsed_before = resume['elapsed_s'] if resume else 0.0
    previous_peak = resume['peak_mb'] if resume else 0.0
    stopped = resume["stopped"] if resume else False
    for epoch in range(tcfg["epochs"] + 1 if stopped else len(history) + 1, tcfg["epochs"] + 1):
        synchronize()
        epoch_started = time.perf_counter()
        model.train()
        accumulated_loss, examples = 0.0, 0
        with Timer() as timer:
            if minibatch:
                perm = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
                for idx in perm.split(minibatch[0]):
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(forward_fn(idx), y[idx])
                    _step(loss, model, opt, tcfg.get("grad_clip"))
                    if sched is not None:
                        sched.step()
                    accumulated_loss += loss.detach().item() * len(idx)
                    examples += len(idx)
            else:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(forward_fn(None)[train_idx], y[train_idx])
                _step(loss, model, opt, tcfg.get("grad_clip"))
                if sched is not None:
                    sched.step()
                accumulated_loss, examples = loss.detach().item() * len(train_idx), len(train_idx)
        epoch_times.append(timer.elapsed)
        train_history.append(accumulated_loss / examples)
        lr_history.append(opt.param_groups[0]['lr'])
        val = evaluate_node(model, forward_fn, y, val_idx, eval_batch)
        history.append(val)
        if val > best["val"]:
            best.update(val=val, epoch=epoch)
            best_state = _cpu_state(model)
            bad = 0
        else:
            bad += 1
        synchronize()
        epoch_wall_times.append(time.perf_counter() - epoch_started)
        if epoch % tcfg.get("log_every", 10) == 0 or epoch == 1:
            log(f"эпоха {epoch:4d} | loss {loss.item():.4f} | val acc {val:.4f} "
                f"| best {best['val']:.4f} (эпоха {best['epoch']}) | {timer.elapsed:.2f} с")
        stopped = epoch >= tcfg.get('min_epochs', 0) and bad >= tcfg.get("patience", 10**9)
        if checkpoint:
            checkpoint.save(model, opt, sched, 'node', tcfg, dict(epoch=epoch,
                best=best, best_state=best_state, history=history, epoch_times=epoch_times,
                epoch_wall_times=epoch_wall_times, train_history=train_history, lr_history=lr_history,
                elapsed_s=elapsed_before + time.perf_counter() - start,
                peak_mb=max(previous_peak, peak_gpu_mem_mb()), bad=bad, stopped=stopped), final=stopped or epoch == tcfg["epochs"])
        if stopped:
            break
    if best_state is None:
        raise RuntimeError("Нет корректного validation checkpoint")
    model.load_state_dict(best_state)
    test = evaluate_node(model, forward_fn, y, split["test"], eval_batch) if evaluate_test else None
    train = evaluate_node(model, forward_fn, y, train_idx, eval_batch)
    return dict(epochs_run=len(history), best_epoch=best["epoch"], best_val_metric=best["val"],
                test_metric_at_best_val=test, train_metric_at_best_val=train,
                test_evaluations=int(evaluate_test), history=history,
                train_loss_history=train_history, learning_rate_history=lr_history,
                train_time_total_s=elapsed_before + time.perf_counter() - start,
                optimization_time_total_s=float(sum(epoch_times)),
                epoch_wall_times_s=epoch_wall_times,
                time_per_epoch_s=float(np.mean(epoch_times)), peak_gpu_mem_mb=max(previous_peak, peak_gpu_mem_mb()))

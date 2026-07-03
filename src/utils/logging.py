"""Logging utilities: a console+file logger plus JEPA-style metric helpers.

Mirrors the I-JEPA_3D_OCT logging conventions (CSVLogger, AverageMeter, gpu_timer,
grad_logger) and adds setup_logging / get_logger so every run is timestamped to the
console and captured under logs/. Standalone: imports only the standard library (and
torch lazily), so it can be loaded in isolation from the rest of the package when the
vendored V-JEPA repo owns the `src` import namespace.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = "logs"


def setup_logging(name: str = "cwlab", log_dir: str = DEFAULT_LOG_DIR,
                  level: int = logging.INFO, to_file: bool = True) -> logging.Logger:
    """Configure and return a logger writing to stdout and (optionally) a run log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{name}_{stamp}.log")
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("logging to %s", path)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # live output under tee / pipes
    except (AttributeError, ValueError):
        pass
    return logger


def get_logger(name: str = "cwlab", **kwargs) -> logging.Logger:
    """Return a logger, configuring it on first use."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name, **kwargs)
    return logger


class CSVLogger:
    """Append one row per call to a CSV file, writing a header if the file is new."""

    def __init__(self, log_file: str, *columns: str) -> None:
        self.log_file = log_file
        self.columns = columns
        parent = os.path.dirname(log_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as handle:
                csv.writer(handle).writerow(columns)

    def log(self, *values) -> None:
        with open(self.log_file, "a", newline="") as handle:
            csv.writer(handle).writerow(values)


class AverageMeter:
    """Track a running average and the most recent value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def gpu_timer(fn):
    """Time a zero-arg callable with CUDA events (ms); fall back to wall clock on CPU."""
    import torch

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end)

    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000.0


def grad_logger(named_params):
    """Return per-parameter gradient stats (name, grad_mean, grad_max); skip grad-less params."""
    stats = []
    for name, param in named_params:
        if param.grad is not None:
            grad = param.grad.data
            stats.append({
                "name": name,
                "grad_mean": grad.abs().mean().item(),
                "grad_max": grad.abs().max().item(),
            })
    return stats

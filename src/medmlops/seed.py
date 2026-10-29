"""Reproducibility helpers: seeding, device selection and worker init."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # pragma: no cover - benchmark mode is opt-in
        torch.backends.cudnn.benchmark = True


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"{name} ({total:.1f} GB VRAM)"
    return f"CPU ({os.cpu_count()} threads)"


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - runs in worker procs
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

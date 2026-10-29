"""Image statistics shared by the training pipeline and drift monitoring."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

FEATURE_NAMES = ("brightness", "contrast", "entropy", "saturation", "colorfulness")
N_BINS = 16


def sample_indices(n: int, limit: int, seed: int = 42) -> np.ndarray:
    """Deterministic random subset of ``range(n)`` for drift baselines.

    Taking the first ``limit`` rows biases the baseline whenever dataset ordering
    correlates with image appearance, so sample across the whole split instead.
    """
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))


def image_entropy(gray: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0, 255))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def compute_image_features(image: np.ndarray) -> dict[str, float]:
    """Compute interpretable, cheap image statistics used for drift detection.

    Args:
        image: HxWx3 array in any numeric range; uint8 inputs are treated as 0-255.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max(initial=0) > 1.0 else (
            np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        )

    rgb = arr.astype(np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)

    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "entropy": image_entropy(gray * 255.0),
        "saturation": float(sat.mean()),
        "colorfulness": colorfulness,
    }


def build_reference_stats(images: Iterable[np.ndarray], dataset: str, sample_limit: int = 600) -> dict:
    """Build per-feature histograms + pixel moments for drift baselines."""
    collected: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    pixels = 0
    for index, image in enumerate(images):
        if index >= sample_limit:
            break
        feats = compute_image_features(image)
        for name in FEATURE_NAMES:
            collected[name].append(feats[name])
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        rgb = arr[..., :3].astype(np.float32) / 255.0
        channel_sum += rgb.reshape(-1, 3).sum(axis=0)
        channel_sq += (rgb.reshape(-1, 3) ** 2).sum(axis=0)
        pixels += rgb.shape[0] * rgb.shape[1]

    features: dict[str, dict] = {}
    for name, values in collected.items():
        values_arr = np.asarray(values, dtype=np.float64)
        counts, edges = np.histogram(values_arr, bins=N_BINS)
        freq = (counts / max(counts.sum(), 1)).tolist()
        features[name] = {
            "mean": float(values_arr.mean()),
            "std": float(values_arr.std()),
            "p05": float(np.percentile(values_arr, 5)),
            "p95": float(np.percentile(values_arr, 95)),
            "bin_edges": [float(e) for e in edges],
            "expected_freq": [float(f) for f in freq],
        }

    mean = channel_sum / max(pixels, 1)
    var = np.maximum(channel_sq / max(pixels, 1) - mean**2, 1e-12)
    return {
        "dataset": dataset,
        "n_samples": len(collected["brightness"]),
        "features": features,
        "channel_mean": mean.tolist(),
        "channel_std": np.sqrt(var).tolist(),
    }


def save_reference_stats(stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load_reference_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

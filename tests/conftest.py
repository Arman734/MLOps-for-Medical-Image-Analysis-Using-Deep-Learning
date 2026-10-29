"""Shared test fixtures: deterministic images and tiny models on CPU."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture()
def rgb_image(rng) -> Image.Image:
    return Image.fromarray(rng.integers(0, 255, size=(80, 96, 3), dtype=np.uint8))


@pytest.fixture()
def feature_reference(rng) -> dict:
    from medmlops.data.common import build_reference_stats

    images = [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(40)]
    return build_reference_stats(images, dataset="test", sample_limit=40)

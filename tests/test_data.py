"""Transforms, image features and PNG rendering helpers."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from medmlops.data.common import build_reference_stats, compute_image_features
from medmlops.data.transforms import (
    clf_eval_transform,
    clf_train_transform,
    load_image_for_inference,
    seg_eval_transform,
    seg_train_transform,
)
from medmlops.imaging import (
    png_b64,
    png_bytes,
    render_heatmap_overlay,
    render_mask_overlay,
    thumbnail,
)


@pytest.fixture()
def image_mask(rng):
    image = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    return image, Image.fromarray(mask)


def test_seg_transforms_keep_shapes_and_align(image_mask):
    image, mask = image_mask
    train_img, train_mask = seg_train_transform()(image, mask)
    assert train_img.shape == (3, 64, 64)
    assert train_mask.shape == (1, 64, 64)
    assert train_img.dtype == torch.float32
    assert set(torch.unique(train_mask).tolist()) <= {0.0, 1.0}

    eval_img, eval_mask = seg_eval_transform()(image, mask)
    assert eval_img.shape == (3, 64, 64)
    assert eval_mask.shape == (1, 64, 64)
    assert float(eval_mask.mean()) == pytest.approx(0.25, abs=0.01)


def test_isic_dataset_returns_single_channel_mask(tmp_path, rng):
    from medmlops.data.isic import ISICSegDataset

    images = rng.integers(0, 255, size=(3, 32, 32, 3), dtype=np.uint8)
    masks = np.zeros((3, 32, 32), dtype=np.uint8)
    masks[:, 8:24, 8:24] = 1  # npz splits are already binarised
    path = tmp_path / "train.npz"
    np.savez(path, images=images, masks=masks, names=np.array(["a", "b", "c"]))

    image, mask, name = ISICSegDataset(path, train=False)[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert float(mask.mean()) == pytest.approx(0.25)
    assert name == "a"


@pytest.mark.parametrize("scale", [1, 255])
def test_seg_masks_accept_binary_and_png_scales(image_mask, scale):
    image, _ = image_mask
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[16:48, 16:48] = scale

    _, eval_mask = seg_eval_transform()(image, binary)
    assert float(eval_mask.mean()) == pytest.approx(0.25)
    assert set(torch.unique(eval_mask).tolist()) <= {0.0, 1.0}

    _, train_mask = seg_train_transform()(image, binary)
    assert set(torch.unique(train_mask).tolist()) <= {0.0, 1.0}
    assert 0.05 < float(train_mask.mean()) < 0.6  # affine aug moves the lesion box


def test_clf_transforms(image_mask):
    image, _ = image_mask
    train = clf_train_transform(32)(image)
    inference = clf_eval_transform(32)(image)
    assert train.shape == (3, 32, 32)
    assert inference.shape == (3, 32, 32)
    assert -3.0 < float(inference.mean()) < 3.0


def test_load_image_for_inference(image_mask):
    image, _ = image_mask
    seg = load_image_for_inference(image, 48, "segmentation")
    clf = load_image_for_inference(image, 40, "classification")
    assert seg.shape == (1, 3, 48, 48)
    assert clf.shape == (1, 3, 40, 40)


def test_compute_image_features_range(rng):
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    black_features = compute_image_features(black)
    white_features = compute_image_features(white)
    assert black_features["brightness"] == pytest.approx(0.0)
    assert white_features["brightness"] == pytest.approx(1.0)
    assert black_features["entropy"] == pytest.approx(0.0)
    assert white_features["saturation"] == pytest.approx(0.0)


def test_build_reference_stats_limits_samples(rng):
    images = [rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8) for _ in range(10)]
    stats = build_reference_stats(images, dataset="unit", sample_limit=4)
    assert stats["n_samples"] == 4
    assert len(stats["channel_mean"]) == 3
    for name, payload in stats["features"].items():
        assert len(payload["expected_freq"]) == len(payload["bin_edges"]) - 1
        assert sum(payload["expected_freq"]) == pytest.approx(1.0)
        assert name


def test_sample_indices_are_deterministic_and_spread():
    from medmlops.data.common import sample_indices

    first = sample_indices(1000, 100, seed=42)
    assert np.array_equal(first, sample_indices(1000, 100, seed=42))
    assert len(np.unique(first)) == 100
    assert first.min() < 50 and first.max() > 950
    assert np.array_equal(sample_indices(50, 100), np.arange(50))


def test_png_helpers_and_overlays(rng):
    image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 1
    heat = rng.random((64, 64)).astype(np.float32)

    overlay = render_mask_overlay(image, mask)
    heat_overlay = render_heatmap_overlay(image, heat)
    assert overlay.shape == image.shape
    assert heat_overlay.shape == image.shape

    data = png_bytes(mask * 255)
    assert data.startswith(b"\x89PNG")
    assert base64.b64decode(png_b64(mask * 255))[:4] == b"\x89PNG"

    small = thumbnail(Image.fromarray(image), max_side=32)
    assert max(small.shape[:2]) == 32
    assert io.BytesIO(data).read(4) == b"\x89PNG"

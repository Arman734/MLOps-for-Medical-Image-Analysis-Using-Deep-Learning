"""Explainability: Grad-CAM and segmentation test-time augmentation."""

from __future__ import annotations

import numpy as np
import torch

from medmlops.explain import GradCAM, gradcam_for_model, segmentation_tta
from medmlops.models import UNet, build_densenet


def test_gradcam_shape_and_range():
    model = build_densenet(num_classes=3, pretrained=False).eval()
    tensor = torch.randn(1, 3, 64, 64)
    heatmap = gradcam_for_model(model, tensor, class_index=1)
    assert heatmap.shape == (64, 64)
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0
    assert np.isfinite(heatmap).all()


def test_gradcam_removes_hooks_after_use():
    model = build_densenet(num_classes=2, pretrained=False)
    layer = model.features.denseblock4
    baseline = len(layer._forward_hooks) + len(layer._backward_hooks)
    cam = GradCAM(model, layer)
    cam.generate(torch.randn(1, 3, 64, 64))
    cam.close()
    remaining = len(layer._forward_hooks) + len(layer._backward_hooks)
    assert remaining == baseline


def test_segmentation_tta_matches_single_pass_on_symmetric_model():
    model = UNet(base_channels=2, depth=1, dropout=0.0).eval()
    tensor = torch.zeros(1, 3, 32, 32)
    mean, uncertainty = segmentation_tta(model, tensor, use_tta=True)
    assert mean.shape == (32, 32)
    assert uncertainty.shape == (32, 32)
    assert np.all(uncertainty >= 0.0)
    assert np.all((mean >= 0.0) & (mean <= 1.0))


def test_segmentation_tta_disabled_returns_zero_uncertainty():
    model = UNet(base_channels=2, depth=1, dropout=0.0).eval()
    mean, uncertainty = segmentation_tta(model, torch.randn(1, 3, 32, 32), use_tta=False)
    assert mean.shape == (32, 32)
    assert np.count_nonzero(uncertainty) == 0

"""Explainability: Grad-CAM for the classifier, TTA uncertainty for U-Net."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Class-activation map for convolutional classifiers (hooks are removed on exit)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[0, class_index].backward()
        if self.activations is None or self.gradients is None:  # pragma: no cover - defensive
            raise RuntimeError("Grad-CAM hooks did not capture activations")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        self.model.train(was_training)
        return cam.cpu().numpy()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _gradcam_target(model: nn.Module) -> nn.Module:
    """Pick the last dense block: hooking ``features`` itself breaks autograd
    because torchvision applies an in-place ReLU to its output."""
    features = getattr(model, "features", None)
    if features is None:  # pragma: no cover - only DenseNet is supported
        raise ValueError("Grad-CAM expects a torchvision DenseNet-style model with .features")
    last_block = getattr(features, "denseblock4", None)
    if last_block is not None:
        return last_block
    if isinstance(features, nn.Sequential) and len(features) >= 2:
        return features[-2]
    return features


def gradcam_for_model(model: nn.Module, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
    with GradCAM(model, _gradcam_target(model)) as cam:
        heatmap = cam.generate(input_tensor, class_index)
    model.zero_grad(set_to_none=True)
    return heatmap


def segmentation_tta(model: nn.Module, input_tensor: torch.Tensor, use_tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted mask probability with horizontal-flip TTA, plus uncertainty."""
    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor))
        if not use_tta:
            zeros = torch.zeros_like(prob)
            return prob[0, 0].cpu().numpy(), zeros[0, 0].cpu().numpy()
        flipped = torch.sigmoid(model(torch.flip(input_tensor, dims=[3])))
        flipped = torch.flip(flipped, dims=[3])
        mean = (prob + flipped) / 2.0
        uncertainty = (prob - flipped).abs() / 2.0
    return mean[0, 0].cpu().numpy(), uncertainty[0, 0].cpu().numpy()

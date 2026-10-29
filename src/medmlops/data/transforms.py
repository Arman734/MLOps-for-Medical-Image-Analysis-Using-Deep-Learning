"""Shared torchvision v2 transforms matching the preprocessing pipeline.

Transforms are implemented as picklable callables so they can be used with
DataLoader worker processes on every platform (Windows spawn included).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_NORMALIZE = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def _binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Accept masks in {0, 1} (npz splits) or {0, 255} (raw PNGs); return float {0, 1}."""
    threshold = 0.5 if float(mask.max()) <= 1.0 else 127.0
    return (mask > threshold).float()


class SegTrainTransform:
    """Synchronised geometric augmentation + image-only photometric jitter."""

    def __init__(self) -> None:
        self.geometry = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
            ]
        )
        self.photometric = v2.Compose(
            [
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __call__(self, image, mask):
        image, mask = self.geometry(image, mask)
        image = self.photometric(image)
        return _NORMALIZE(image), _binarize_mask(mask)


class SegEvalTransform:
    def __call__(self, image, mask=None):
        image = v2.ToImage()(image)
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = _NORMALIZE(image)
        if mask is None:
            return image
        return image, _binarize_mask(v2.ToImage()(mask))


def seg_train_transform() -> SegTrainTransform:
    return SegTrainTransform()


def seg_eval_transform() -> SegEvalTransform:
    return SegEvalTransform()


def clf_train_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=8, fill=0),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            v2.ToDtype(torch.float32, scale=True),
            _NORMALIZE,
        ]
    )


def clf_eval_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(image_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            _NORMALIZE,
        ]
    )


def load_image_for_inference(image, image_size: int, task: str) -> torch.Tensor:
    """Prepare a PIL image for inference exactly like the evaluation pipeline."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if task == "segmentation":
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return seg_eval_transform()(image).unsqueeze(0)
    return clf_eval_transform(image_size)(image).unsqueeze(0)

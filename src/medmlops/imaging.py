"""Image rendering helpers shared by the API, CLI and training plots."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

MASK_RGB = np.array([34, 211, 238], dtype=np.float32)  # cyan
GT_RGB = np.array([251, 113, 133], dtype=np.float32)


def to_pil(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    to_pil(array).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def png_b64(array: np.ndarray) -> str:
    return base64.b64encode(png_bytes(array)).decode("ascii")


def render_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=None) -> np.ndarray:
    """Blend a binary mask over an RGB image with a soft edge."""
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max(initial=0) <= 1.0:
        rgb = rgb * 255.0
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    m = (m > 0.5).astype(np.float32)
    tint = np.asarray(color if color is not None else MASK_RGB, dtype=np.float32)
    out = rgb.copy()
    out[m > 0] = (1 - alpha) * rgb[m > 0] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_map(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(heatmap.astype(np.float32), mode="F").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def render_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55, colormap: str = "magma") -> np.ndarray:
    """Overlay a 0-1 saliency map on an RGB image."""
    from matplotlib import colormaps

    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max(initial=0) <= 1.0:
        rgb = rgb * 255.0
    cam = np.asarray(heatmap, dtype=np.float32)
    cam = np.clip(cam, 0, 1)
    cam = _resize_map(cam, rgb.shape[:2])
    colored = colormaps[colormap](cam)[..., :3] * 255.0
    out = (1 - alpha * cam[..., None]) * rgb + (alpha * cam[..., None]) * colored
    return np.clip(out, 0, 255).astype(np.uint8)


def render_uncertainty(image: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    return render_heatmap_overlay(image, uncertainty, alpha=0.55, colormap="turbo")


def side_by_side(*panels: tuple[np.ndarray, str]) -> np.ndarray:
    """Horizontally concatenate RGB panels with thin separators."""
    arrays = [p[0] for p in panels]
    height = min(a.shape[0] for a in arrays)
    resized = [np.asarray(Image.fromarray(a.astype(np.uint8)).resize((int(a.shape[1] * height / a.shape[0]), height))) for a in arrays]
    separator = np.full((height, 4, 3), 11, dtype=np.uint8)
    out = resized[0]
    for panel in resized[1:]:
        out = np.concatenate([out, separator, panel], axis=1)
    return out


def thumbnail(image: Image.Image, max_side: int = 512) -> np.ndarray:
    rgb = image.convert("RGB")
    scale = max_side / max(rgb.size)
    if scale < 1:
        rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8)

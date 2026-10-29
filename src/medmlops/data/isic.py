"""ISIC 2018 skin-lesion segmentation: download, preprocess, splits, Dataset.

The pipeline consumes the widely mirrored ISIC 2018 Task 1 (segmentation) images
from Hugging Face by default, and can fall back to the official 11 GB archive.
Every split is materialised as a compressed ``.npz`` so training never pays the
cost of decoding 767x1022 PNGs again.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import PROJECT_ROOT
from .common import build_reference_stats, sample_indices, save_reference_stats
from .transforms import seg_eval_transform, seg_train_transform

HF_REPO = "sergiomadrid/isic_segmentation"
OFFICIAL_ZIPS = {
    "images": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
    "masks": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
}


@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path


def _raw_dir(cfg) -> Path:
    return PROJECT_ROOT / cfg.data.raw_root / "isic2018"


def _processed_dir(cfg) -> Path:
    return PROJECT_ROOT / cfg.data.processed_root / "isic2018"


def _download_official(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for key, url in OFFICIAL_ZIPS.items():
        archive = dest / f"{key}.zip"
        if not archive.exists():
            print(f"[isic] downloading official archive ({key}) - this is ~11 GB, resumable")
            _stream_download(url, archive)
        target = dest / key
        if target.exists():
            continue
        print(f"[isic] extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest / f"_extract_{key}")
        extracted = [p for p in (dest / f"_extract_{key}").rglob("*") if p.is_file()]
        target.mkdir(parents=True, exist_ok=True)
        for path in extracted:
            shutil.move(str(path), str(target / path.name))
        shutil.rmtree(dest / f"_extract_{key}", ignore_errors=True)


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        last = time.time()
        while True:
            data = response.read(chunk)
            if not data:
                break
            fh.write(data)
            seen += len(data)
            if time.time() - last > 5:
                pct = 100 * seen / total if total else 0
                print(f"[isic]   {seen / 1e9:.2f} GB ({pct:.1f}%)")
                last = time.time()
    tmp.replace(dest)


def download(cfg, source: str = "hf", force: bool = False) -> Path:
    """Fetch ISIC 2018 images + masks into ``data/raw/isic2018``."""
    dest = _raw_dir(cfg)
    marker = dest / "download_complete.json"
    if marker.exists() and not force:
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    if source == "hf":
        try:
            from huggingface_hub import snapshot_download

            print(f"[isic] fetching mirror {HF_REPO} from Hugging Face")
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=str(dest),
                allow_patterns=["images_segmentation/*", "ground_truth/*", "isic_segmentation.yml"],
                max_workers=8,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[isic] mirror unavailable ({exc}); falling back to official archives")
            _download_official(dest)
    elif source == "official":
        _download_official(dest)
    else:
        raise ValueError(f"unknown ISIC source: {source}")

    pairs = collect_pairs(dest)
    if len(pairs) < 50:
        raise RuntimeError(f"ISIC download incomplete: only {len(pairs)} image/mask pairs found in {dest}")
    marker.write_text(
        json.dumps({"source": source, "pairs": len(pairs), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2),
        encoding="utf-8",
    )
    return dest


def collect_pairs(raw_dir: Path) -> list[Pair]:
    """Pair images with their ground-truth masks (metadata YAML preferred)."""
    image_dir = raw_dir / "images_segmentation"
    mask_dir = raw_dir / "ground_truth"
    if not image_dir.exists():
        image_dir = raw_dir / "images"
    if not mask_dir.exists():
        mask_dir = raw_dir / "masks"
    if not image_dir.exists():
        return []

    pairs: list[Pair] = []
    metadata = raw_dir / "isic_segmentation.yml"
    names: list[str] = []
    if metadata.exists():
        import yaml

        meta = yaml.safe_load(metadata.read_text(encoding="utf-8")) or {}
        for entry in meta.get("images", []):
            names.append(Path(entry["location"]).stem)
        if not names:
            names = []
    if not names:
        names = sorted(p.stem for p in image_dir.glob("*.jpg"))
        names += sorted(p.stem for p in image_dir.glob("*.png"))

    for name in sorted(set(names)):
        image = next((image_dir / f"{name}{ext}" for ext in (".jpg", ".png", ".jpeg") if (image_dir / f"{name}{ext}").exists()), None)
        mask = mask_dir / f"{name}_segmentation.png"
        if image is None or not mask.exists():
            continue
        pairs.append(Pair(image=image, mask=mask))
    return pairs


def _resize_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)


def _resize_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        mask = im.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(mask) > 127).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_indices(n: int, seed: int, val_fraction: float, test_fraction: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return {
        "test": np.sort(order[:n_test]),
        "val": np.sort(order[n_test : n_test + n_val]),
        "train": np.sort(order[n_test + n_val :]),
    }


def prepare(cfg, force: bool = False) -> dict:
    """Build uint8 image/mask arrays per split plus drift reference statistics."""
    raw = _raw_dir(cfg)
    pairs = collect_pairs(raw) if raw.exists() else []
    if not pairs:
        download(cfg, source=cfg.data.get("source", "hf"), force=force)
        pairs = collect_pairs(raw)
    if not pairs:
        raise RuntimeError("no ISIC image/mask pairs found; run `medmlops data download` first")

    size = int(cfg.data.image_size)
    seed = int(cfg.seed)
    max_samples = cfg.data.get("max_samples")
    if max_samples:
        pairs = pairs[: int(max_samples)]

    out_dir = _processed_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("image_size") == size and manifest.get("seed") == seed and manifest.get("pairs") == len(pairs):
            print(f"[isic] processed data already present ({manifest['pairs']} pairs)")
            return manifest

    splits = split_indices(len(pairs), seed, float(cfg.data.val_fraction), float(cfg.data.test_fraction))
    print(f"[isic] processing {len(pairs)} pairs at {size}x{size} -> {out_dir}")

    images = np.stack([_resize_image(p.image, size) for p in pairs])
    masks = np.stack([_resize_mask(p.mask, size) for p in pairs])
    names = np.array([p.image.stem for p in pairs])

    manifest: dict = {
        "dataset": "isic2018",
        "source": str(raw),
        "pairs": len(pairs),
        "image_size": size,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": {},
        "files": {},
        "lesion_area_mean": float(masks.mean()),
    }
    for split, idx in splits.items():
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, images=images[idx], masks=masks[idx], names=names[idx])
        manifest["splits"][split] = {"count": int(idx.size), "file": path.name}
        manifest["files"][path.name] = _sha256(path)
        print(f"[isic]   {split}: {idx.size} samples ({path.stat().st_size / 1e6:.0f} MB)")

    train_idx = splits["train"]
    baseline_idx = train_idx[sample_indices(len(train_idx), 600, seed)]
    stats = build_reference_stats((images[i] for i in baseline_idx), dataset="isic2018")
    save_reference_stats(stats, out_dir / "reference_stats.json")
    manifest["reference_stats"] = "reference_stats.json"

    _save_preview(images, masks, out_dir / "preview.png")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _save_preview(images: np.ndarray, masks: np.ndarray, path: Path, n: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    for col in range(n):
        index = int(col * (len(images) - 1) / max(n - 1, 1))
        axes[0, col].imshow(images[index])
        axes[0, col].axis("off")
        axes[1, col].imshow(masks[index], cmap="magma", vmin=0, vmax=1)
        axes[1, col].axis("off")
    axes[0, 0].set_title("image", loc="left", fontsize=10)
    axes[1, 0].set_title("ground truth", loc="left", fontsize=10)
    fig.suptitle("ISIC 2018 - lesion segmentation sample", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


class ISICSegDataset(torch.utils.data.Dataset):
    """Segmentation dataset backed by a preprocessed ``.npz`` split."""

    def __init__(self, npz_path: str | Path, train: bool = False) -> None:
        bundle = np.load(Path(npz_path))
        self.images = bundle["images"]
        self.masks = bundle["masks"]
        self.names = bundle["names"]
        self.transform = seg_train_transform() if train else seg_eval_transform()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index]
        mask = self.masks[index]
        image_t, mask_t = self.transform(image, mask)
        return image_t, mask_t, str(self.names[index])


def load_split(cfg, split: str, train: bool | None = None) -> ISICSegDataset:
    path = _processed_dir(cfg) / f"{split}.npz"
    if not path.exists():
        prepare(cfg)
    if train is None:
        train = split == "train"
    return ISICSegDataset(path, train=train)


def dataset_summary(cfg) -> dict:
    manifest_path = _processed_dir(cfg) / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"dataset": "isic2018", "status": "not prepared", "splits": {}}

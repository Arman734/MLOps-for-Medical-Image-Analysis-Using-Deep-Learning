"""MedMNIST classification data (PneumoniaMNIST by default, any MedMNIST works)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import PROJECT_ROOT
from .common import build_reference_stats, sample_indices, save_reference_stats
from .transforms import clf_eval_transform, clf_train_transform

SUPPORTED = {
    "pneumoniamnist": "PneumoniaMNIST",
    "dermamnist": "DermaMNIST",
    "pathmnist": "PathMNIST",
    "bloodmnist": "BloodMNIST",
    "breastmnist": "BreastMNIST",
    "octmnist": "OCTMNIST",
}


def _raw_dir(cfg) -> Path:
    return PROJECT_ROOT / cfg.data.raw_root / "medmnist"


def _processed_dir(cfg) -> Path:
    name = cfg.data.name
    return PROJECT_ROOT / cfg.data.processed_root / "medmnist" / f"{name}_{cfg.data.image_size}"


def _base_dataset(cfg, split: str):
    import medmnist

    name = str(cfg.data.name).lower()
    if name not in SUPPORTED:
        raise ValueError(f"unsupported MedMNIST dataset '{name}'; choose one of {sorted(SUPPORTED)}")
    cls = getattr(medmnist, SUPPORTED[name])
    root = _raw_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    return cls(
        split=split,
        root=str(root),
        download=bool(cfg.data.get("download", True)),
        size=int(cfg.data.image_size),
        as_rgb=True,
    )


class MedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform) -> None:
        self.base = base
        self.transform = transform
        self.labels = np.asarray(base.labels).reshape(-1).astype(np.int64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, _ = self.base[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = int(self.labels[index])
        return self.transform(image), label, f"{self.base.__class__.__name__.lower()}_{index:05d}"


def class_names(cfg) -> list[str]:
    import medmnist

    info = medmnist.INFO[str(cfg.data.name).lower()]
    labels = info["label"]
    return [labels[str(i)] for i in sorted(int(k) for k in labels)]


def prepare(cfg, force: bool = False) -> dict:
    """Download MedMNIST npz, record the class balance and drift reference stats."""
    out_dir = _processed_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    names = class_names(cfg)
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("class_names") == names:
            return manifest

    splits: dict[str, dict] = {}
    train_images: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        base = _base_dataset(cfg, split)
        labels = np.asarray(base.labels).reshape(-1).astype(int)
        counts = np.bincount(labels, minlength=len(names)).tolist()
        splits[split] = {"count": int(len(base)), "class_counts": counts}
        if split == "train":
            for index in sample_indices(len(base), 500, int(cfg.seed)):
                image = base[index][0]
                train_images.append(np.asarray(image.convert("RGB")))

    manifest = {
        "dataset": cfg.data.name,
        "image_size": int(cfg.data.image_size),
        "class_names": names,
        "splits": splits,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_stats": "reference_stats.json",
    }
    stats = build_reference_stats(train_images, dataset=str(cfg.data.name))
    save_reference_stats(stats, out_dir / "reference_stats.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(cfg, split: str, train: bool | None = None) -> MedMNISTDataset:
    if train is None:
        train = split == "train"
    transform = clf_train_transform(int(cfg.data.image_size)) if train else clf_eval_transform(
        int(cfg.data.image_size)
    )
    return MedMNISTDataset(_base_dataset(cfg, split), transform)


def dataset_summary(cfg) -> dict:
    manifest_path = _processed_dir(cfg) / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"dataset": cfg.data.name, "status": "not prepared", "splits": {}}


def reference_stats_path(cfg) -> Path:
    return _processed_dir(cfg) / "reference_stats.json"

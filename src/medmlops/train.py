"""Training entry points: U-Net segmentation and DenseNet classification.

Both tasks share the same machinery: AMP, cosine schedule with warmup,
early stopping on a validation metric, MLflow logging, checkpoint export and
automatic champion promotion in the MLflow model registry.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from . import tracking
from .checkpoint import new_meta, save_checkpoint
from .config import PROJECT_ROOT, flatten, load_config
from .data import isic, medmnist
from .metrics import best_threshold, classification_metrics, segmentation_metrics
from .models import build_model, count_parameters, freeze_backbone
from .plots import (
    confusion_matrix_plot,
    probability_distribution,
    roc_curve_plot,
    seg_prediction_grid,
    training_curves,
)
from .report import classification_card, segmentation_card
from .seed import describe_device, pick_device, seed_everything, worker_init_fn


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _loader(dataset, cfg, *, train: bool, device: torch.device) -> DataLoader:
    workers = int(cfg.data.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=train,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=train and len(dataset) > int(cfg.training.batch_size),
        worker_init_fn=worker_init_fn if workers > 0 else None,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )


def _optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )


def _scheduler(optimizer, total_steps: int, warmup_frac: float = 0.03):
    warmup = max(int(total_steps * warmup_frac), 1)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def _step(
    model: nn.Module,
    batch,
    criterion: nn.Module,
    device: torch.device,
    optimizer,
    scaler,
    scheduler,
    grad_clip: float,
    amp: bool,
):
    inputs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(device, amp):
        logits = model(inputs)
        loss = criterion(logits, targets)
    scaler.scale(loss).backward()
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.detach()


def _finish_run(cfg, metrics: dict, artifacts: list[Path], model, meta: dict, metric_key: str, card: str) -> dict:
    import mlflow

    run = mlflow.active_run()
    run_id = run.info.run_id if run else "unknown"

    # Log final (best-checkpoint) metrics one step past the last epoch so MLflow's
    # run-level summary reflects the restored model, not the last epoch's weights.
    final_step = int(metrics.get("epochs_trained", 0)) + 1
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value == value:
            mlflow.log_metric(key, float(value), step=final_step)
    for artifact in artifacts:
        mlflow.log_artifact(str(artifact), artifact_path="plots" if artifact.suffix == ".png" else None)

    metrics_path = PROJECT_ROOT / "metrics" / f"{cfg.experiment}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mlflow.log_text(card, "model_card.md")

    meta = {**meta, "run_id": run_id, "metrics": metrics}
    checkpoint = save_checkpoint(PROJECT_ROOT / cfg.output.model_dir / cfg.output.name, model, meta)
    mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
    mlflow.log_artifact(str(checkpoint.with_suffix(".json")), artifact_path="checkpoints")

    if bool(cfg.mlflow.get("log_model", True)):
        input_example = torch.zeros(1, 3, int(cfg.data.image_size), int(cfg.data.image_size))
        registered = tracking.register_model(model, cfg, input_example, card, metrics, run_id)
        mlflow.set_tag("registry_version", str(registered["version"]))
        tracking.maybe_promote(cfg, run_id, metric_key, float(metrics.get(metric_key, float("-inf"))))

    result = {"run_id": run_id, "metrics": metrics, "checkpoint": str(checkpoint), "meta": meta}
    (PROJECT_ROOT / "artifacts" / "last_run.json").parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts" / "last_run.json").write_text(
        json.dumps({"experiment": cfg.experiment, "run_id": run_id, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    return result


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 2.0, bce_weight: float = 0.5, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
        prob = torch.sigmoid(logits).flatten(1)
        flat_target = target.flatten(1)
        intersection = (prob * flat_target).sum(dim=1)
        denom = prob.sum(dim=1) + flat_target.sum(dim=1)
        dice_loss = 1.0 - ((2 * intersection + 1.0) / (denom + 1.0)).mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device, threshold: float = 0.5, collect_probs: bool = False, max_collect: int = 8):
    model.eval()
    total_loss, batches = 0.0, 0
    collected = {"images": [], "masks": [], "probs": []} if collect_probs else None
    all_probs, all_targets = [], []
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, masks))
        batches += 1
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_targets.append(masks.cpu())
        if collect_probs and len(collected["images"]) < max_collect:
            keep = min(4, images.size(0), max_collect - len(collected["images"]))
            collected["images"].append(images[:keep].cpu())
            collected["masks"].append(masks[:keep].cpu())
            collected["probs"].append(probs[:keep].cpu())
    probs_cat = torch.cat(all_probs)
    targets_cat = torch.cat(all_targets)
    metrics = segmentation_metrics(probs_cat >= threshold, targets_cat)
    metrics["loss"] = total_loss / max(batches, 1)
    if collect_probs:
        collected = {k: torch.cat(v).numpy() for k, v in collected.items()}
        return metrics, collected, probs_cat.numpy(), targets_cat.numpy()
    return metrics, probs_cat.numpy(), targets_cat.numpy()


def train_segmentation(cfg, device: torch.device, dagshub: bool | None = None) -> dict:
    import mlflow
    from tqdm import tqdm

    tracking.init_tracking(cfg, dagshub)
    manifest = isic.prepare(cfg)
    train_ds = isic.load_split(cfg, "train")
    val_ds = isic.load_split(cfg, "val")
    test_ds = isic.load_split(cfg, "test")

    train_loader = _loader(train_ds, cfg, train=True, device=device)
    val_loader = _loader(val_ds, cfg, train=False, device=device)
    test_loader = _loader(test_ds, cfg, train=False, device=device)

    model = build_model(cfg).to(device)
    if bool(cfg.model.get("freeze_backbone", False)):
        freeze_backbone(model)
    n_params = count_parameters(model)

    criterion = DiceBCELoss(
        pos_weight=float(cfg.training.get("pos_weight", 2.0)),
        bce_weight=float(cfg.training.get("bce_weight", 0.5)),
        dice_weight=float(cfg.training.get("dice_weight", 1.0)),
    )
    optimizer = _optimizer(model, cfg)
    epochs = int(cfg.training.epochs)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = _scheduler(optimizer, total_steps)
    amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best = {"val_dice": -1.0, "epoch": 0, "state": None}
    patience = int(cfg.training.get("patience", 8))
    stale = 0
    started = time.time()

    run_name = cfg.get("run_name") or f"{cfg.experiment}-{time.strftime('%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in flatten(cfg).items() if v is not None})
        mlflow.set_tags(
            {
                "task": "segmentation",
                "dataset": cfg.data.name,
                "git_commit": tracking.git_commit(),
                "git_dirty": str(tracking.git_dirty()),
                "device": describe_device(device),
                "params": str(n_params),
                "module": "medmlops.train",
            }
        )
        mlflow.log_artifact(str(PROJECT_ROOT / "data" / "processed" / "isic2018" / "manifest.json"), artifact_path="data")

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss, seen = 0.0, 0
            progress = tqdm(train_loader, desc=f"epoch {epoch:02d}/{epochs}", leave=False)
            for step, batch in enumerate(progress, start=1):
                loss = _step(model, batch, criterion, device, optimizer, scaler, scheduler, float(cfg.training.get("grad_clip", 1.0)), amp)
                epoch_loss += float(loss) * batch[0].size(0)
                seen += batch[0].size(0)
                global_step = (epoch - 1) * len(train_loader) + step
                if step % int(cfg.training.get("log_every_n_steps", 20)) == 0:
                    mlflow.log_metric("train_step_loss", float(loss), step=global_step)
                    progress.set_postfix(loss=f"{float(loss):.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

            train_loss = epoch_loss / max(seen, 1)
            val_metrics, _, _ = evaluate_segmentation(model, val_loader, criterion, device)
            lr = scheduler.get_last_lr()[0]
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_iou"].append(val_metrics["iou"])
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
            for key, value in epoch_metrics.items():
                mlflow.log_metric(key, float(value), step=epoch)
            mlflow.log_metric("lr", lr, step=epoch)
            print(
                f"[seg] epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
            )

            if val_metrics["dice"] > best["val_dice"]:
                best = {
                    "val_dice": val_metrics["dice"],
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"[seg] early stopping at epoch {epoch} (best epoch {best['epoch']})")
                    break

        if best["state"] is not None:
            model.load_state_dict(best["state"])
            model.to(device)

        val_metrics, _, val_probs, val_targets = evaluate_segmentation(
            model, val_loader, criterion, device, collect_probs=True, max_collect=4
        )
        threshold = best_threshold(val_probs, val_targets)
        test_metrics, sample, _, _ = evaluate_segmentation(
            model, test_loader, criterion, device, threshold=threshold, collect_probs=True
        )
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        final = {
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
            "val_loss": val_metrics["loss"],
            **test_metrics,
            "threshold": threshold,
            "epochs_trained": float(len(history["epoch"])),
            "best_epoch": float(best["epoch"]),
            "train_seconds": time.time() - started,
            "parameters": float(n_params),
        }

        curves = training_curves(history, PROJECT_ROOT / "artifacts" / "figures" / f"{cfg.experiment}_curves.png", "dice")
        grid = seg_prediction_grid(
            sample["images"], sample["masks"], sample["probs"],
            PROJECT_ROOT / "artifacts" / "figures" / f"{cfg.experiment}_predictions.png", threshold,
        )

        meta = new_meta(
            task="segmentation",
            arch=str(cfg.model.arch),
            model_kwargs={
                "in_channels": int(cfg.model.get("in_channels", 3)),
                "out_channels": int(cfg.model.get("out_channels", 1)),
                "base_channels": int(cfg.model.get("base_channels", 32)),
                "depth": int(cfg.model.get("depth", 4)),
                "dropout": float(cfg.model.get("dropout", 0.1)),
            },
            image_size=int(cfg.data.image_size),
            dataset=str(cfg.data.name),
            run_id="pending",
            git_commit=tracking.git_commit(),
            config_path=str(cfg.get("_config_path", "")),
            threshold=float(threshold),
            class_names=["background", "lesion"],
            normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            dataset_summary=manifest,
            best_epoch=best["epoch"],
        )
        card = segmentation_card(
            dataset=str(cfg.data.name),
            dataset_size={k: v["count"] for k, v in manifest["splits"].items()},
            model_desc=f"U-Net (base {cfg.model.base_channels}, depth {cfg.model.depth}), {n_params / 1e6:.2f}M params",
            metrics={k.removeprefix("test_"): v for k, v in test_metrics.items()},
            threshold=threshold,
            run_id=mlflow.active_run().info.run_id,
            git_commit=tracking.git_commit(),
            device=describe_device(device),
            extra=(
                f"\n- **Validation Dice:** {val_metrics['dice']:.4f}"
                f"\n- **Best epoch:** {best['epoch']} / {len(history['epoch'])}\n"
            ),
        )
        return _finish_run(cfg, final, [curves, grid], model, meta, "val_dice", card)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for images, target, _ in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(target.cpu())
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train_classification(cfg, device: torch.device, dagshub: bool | None = None) -> dict:
    import mlflow
    from tqdm import tqdm

    tracking.init_tracking(cfg, dagshub)
    manifest = medmnist.prepare(cfg)
    names = manifest["class_names"]
    num_classes = len(names)

    train_ds = medmnist.load_split(cfg, "train")
    val_ds = medmnist.load_split(cfg, "val")
    test_ds = medmnist.load_split(cfg, "test")
    train_loader = _loader(train_ds, cfg, train=True, device=device)
    val_loader = _loader(val_ds, cfg, train=False, device=device)
    test_loader = _loader(test_ds, cfg, train=False, device=device)

    model = build_model(cfg).to(device)
    if bool(cfg.model.get("freeze_backbone", False)):
        freeze_backbone(model)
    n_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.get("label_smoothing", 0.0)))
    optimizer = _optimizer(model, cfg)
    epochs = int(cfg.training.epochs)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = _scheduler(optimizer, total_steps)
    amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best = {"val_auc": -1.0, "epoch": 0, "state": None}
    patience = int(cfg.training.get("patience", 4))
    stale = 0
    started = time.time()

    run_name = cfg.get("run_name") or f"{cfg.experiment}-{time.strftime('%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in flatten(cfg).items() if v is not None})
        mlflow.set_tags(
            {
                "task": "classification",
                "dataset": cfg.data.name,
                "git_commit": tracking.git_commit(),
                "git_dirty": str(tracking.git_dirty()),
                "device": describe_device(device),
                "params": str(n_params),
                "module": "medmlops.train",
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss, seen = 0.0, 0
            progress = tqdm(train_loader, desc=f"epoch {epoch:02d}/{epochs}", leave=False)
            for step, batch in enumerate(progress, start=1):
                loss = _step(model, batch, criterion, device, optimizer, scaler, scheduler, float(cfg.training.get("grad_clip", 1.0)), amp)
                epoch_loss += float(loss) * batch[0].size(0)
                seen += batch[0].size(0)
                if step % int(cfg.training.get("log_every_n_steps", 20)) == 0:
                    global_step = (epoch - 1) * len(train_loader) + step
                    mlflow.log_metric("train_step_loss", float(loss), step=global_step)
                    progress.set_postfix(loss=f"{float(loss):.4f}")

            train_loss = epoch_loss / max(seen, 1)
            val_probs, val_labels = predict_probabilities(model, val_loader, device)
            val_metrics = classification_metrics(val_labels, val_probs, names)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["log_loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_auc"].append(val_metrics["auc"])
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["log_loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_auc": val_metrics["auc"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            print(
                f"[clf] epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
            )

            if val_metrics["auc"] > best["val_auc"]:
                best = {"val_auc": val_metrics["auc"], "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"[clf] early stopping at epoch {epoch} (best epoch {best['epoch']})")
                    break

        if best["state"] is not None:
            model.load_state_dict(best["state"])
            model.to(device)

        test_probs, test_labels = predict_probabilities(model, test_loader, device)
        test_metrics = classification_metrics(test_labels, test_probs, names)
        final = {
            **{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))},
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "class_names": names,
            "epochs_trained": float(len(history["epoch"])),
            "best_epoch": float(best["epoch"]),
            "train_seconds": time.time() - started,
            "parameters": float(n_params),
        }

        curves = training_curves(history, PROJECT_ROOT / "artifacts" / "figures" / f"{cfg.experiment}_curves.png", "AUC")
        cm_plot = confusion_matrix_plot(test_metrics["confusion_matrix"], names, PROJECT_ROOT / "artifacts" / "figures" / f"{cfg.experiment}_confusion.png")
        roc = roc_curve_plot(test_labels, test_probs, names, PROJECT_ROOT / "artifacts" / "figures" / f"{cfg.experiment}_roc.png")
        dist = probability_distribution(test_labels, test_probs, names, PROJECT_ROOT / "artifacts" / "figures" / f"{cfg.experiment}_confidence.png")

        meta = new_meta(
            task="classification",
            arch=str(cfg.model.arch),
            model_kwargs={
                "num_classes": num_classes,
                "dropout": float(cfg.model.get("dropout", 0.2)),
            },
            image_size=int(cfg.data.image_size),
            dataset=str(cfg.data.name),
            run_id="pending",
            git_commit=tracking.git_commit(),
            config_path=str(cfg.get("_config_path", "")),
            class_names=names,
            normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            dataset_summary=manifest,
            best_epoch=best["epoch"],
        )
        card = classification_card(
            dataset=str(cfg.data.name),
            dataset_size={k: v["count"] for k, v in manifest["splits"].items()},
            model_desc=f"DenseNet-121 (ImageNet pretrained, {n_params / 1e6:.2f}M params)",
            metrics={k.removeprefix("test_"): v for k, v in test_metrics.items()},
            threshold=None,
            run_id=mlflow.active_run().info.run_id,
            git_commit=tracking.git_commit(),
            device=describe_device(device),
            extra=f"\n- **Classes:** {', '.join(names)}\n- **Validation AUC:** {val_metrics['auc']:.4f}\n- **Best epoch:** {best['epoch']} / {len(history['epoch'])}\n",
        )
        return _finish_run(cfg, final, [curves, cm_plot, roc, dist], model, meta, "val_auc", card)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _apply_overrides(cfg, overrides: list[str]) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got '{override}'")
        key, _, raw = override.partition("=")
        import yaml

        value: Any = yaml.safe_load(raw)
        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {}) if isinstance(node, dict) else getattr(node, part)
        if isinstance(node, dict):
            node[parts[-1]] = value
        else:  # pragma: no cover - defensive
            setattr(node, parts[-1], value)


def train(config_path: str, dagshub: bool | None = None, device: str = "auto", overrides: list[str] | None = None) -> dict:
    cfg = load_config(config_path)
    _apply_overrides(cfg, overrides or [])
    seed_everything(int(cfg.seed))
    torch_device = pick_device(device)
    print(f"[train] {cfg.experiment} on {describe_device(torch_device)}")
    if cfg.task == "segmentation":
        return train_segmentation(cfg, torch_device, dagshub)
    if cfg.task == "classification":
        return train_classification(cfg, torch_device, dagshub)
    raise ValueError(f"unknown task: {cfg.task}")

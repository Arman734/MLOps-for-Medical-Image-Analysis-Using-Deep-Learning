"""Command line interface: data → train → evaluate → predict → serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_config


def _cmd_data(args) -> int:
    from .data import isic, medmnist

    cfg = load_config(args.config)
    if cfg.data.name == "isic2018":
        if args.action == "download":
            raw = isic.download(cfg, source=args.source, force=args.force)
            print(f"[data] downloaded ISIC to {raw}")
            return 0
        manifest = isic.prepare(cfg, force=args.force)
        print(json.dumps({k: manifest[k] for k in ("dataset", "pairs", "image_size", "splits")}, indent=2))
    else:
        manifest = medmnist.prepare(cfg, force=args.force)
        print(json.dumps(manifest, indent=2))
    return 0


def _cmd_train(args) -> int:
    from .train import train

    result = train(args.config, dagshub=args.dagshub, device=args.device, overrides=args.set)
    print(json.dumps({"run_id": result["run_id"], "metrics": result["metrics"], "checkpoint": result["checkpoint"]}, indent=2))
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluate import evaluate_checkpoint

    metrics = evaluate_checkpoint(args.config, checkpoint=args.checkpoint, split=args.split, device=args.device)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_predict(args) -> int:
    from PIL import Image

    from .imaging import png_bytes
    from .predictor import Predictor
    from .seed import pick_device

    predictor = Predictor(device=args.device)
    predictor.refresh()
    image = Image.open(args.image).convert("RGB")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "segmentation":
        result = predictor.segment(image, model_id=args.model, use_tta=not args.no_tta)
        (out_dir / "mask.png").write_bytes(png_bytes(result["mask"] * 255))
        (out_dir / "overlay.png").write_bytes(png_bytes(result["overlay"]))
        summary = {k: v for k, v in result.items() if k in {"model_id", "threshold", "lesion_fraction", "mean_probability", "uncertainty_mean", "latency_ms"}}
    else:
        result = predictor.classify(image, model_id=args.model)
        if result["gradcam"] is not None:
            (out_dir / "gradcam.png").write_bytes(png_bytes(result["gradcam"]))
        summary = {k: v for k, v in result.items() if k in {"model_id", "predicted_label", "confidence", "margin", "latency_ms"}}
        summary["probabilities"] = dict(zip(result["labels"], [round(p, 4) for p in result["probabilities"]], strict=False))
    summary["device"] = pick_device(args.device).type
    summary["outputs"] = str(out_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_registry(args) -> int:
    from . import tracking

    if args.action == "list":
        print(json.dumps(tracking.list_models(), indent=2))
        return 0
    if args.action == "promote":
        cfg = load_config(args.config)
        tracking.init_tracking(cfg, False)
        name, version = tracking.promote_run(cfg, args.run_id)
        print(f"promoted {name} version {version} to alias 'champion'")
        return 0
    if args.action == "champion":
        print(json.dumps(tracking.get_champion(load_config(args.config)), indent=2, default=str))
        return 0
    raise SystemExit(f"unknown registry action: {args.action}")


def _cmd_serve(args) -> int:
    import uvicorn

    sys.path.insert(0, str(PROJECT_ROOT))
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medmlops", description="Reproducible MLOps for medical imaging")
    parser.add_argument("--version", action="version", version="medmlops 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download and prepare datasets")
    data.add_argument("action", choices=["download", "prepare"])
    data.add_argument("--config", required=True)
    data.add_argument("--source", choices=["hf", "official"], default="hf")
    data.add_argument("--force", action="store_true")
    data.set_defaults(func=_cmd_data)

    train_p = sub.add_parser("train", help="train a model and track it with MLflow")
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--set", action="append", default=[], help="override config, e.g. training.epochs=3")
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--dagshub", action="store_true", help="stream the run to DagsHub MLflow")
    train_p.set_defaults(func=_cmd_train)

    ev = sub.add_parser("evaluate", help="evaluate a checkpoint")
    ev.add_argument("--config", required=True)
    ev.add_argument("--checkpoint", default=None)
    ev.add_argument("--split", default="test", choices=["train", "val", "test"])
    ev.add_argument("--device", default="auto")
    ev.set_defaults(func=_cmd_evaluate)

    pred = sub.add_parser("predict", help="run inference on a single image")
    pred.add_argument("--task", required=True, choices=["segmentation", "classification"])
    pred.add_argument("--image", required=True)
    pred.add_argument("--model", default=None)
    pred.add_argument("--out", default="artifacts/predictions")
    pred.add_argument("--device", default="auto")
    pred.add_argument("--no-tta", action="store_true")
    pred.set_defaults(func=_cmd_predict)

    reg = sub.add_parser("registry", help="inspect / promote registered models")
    reg.add_argument("action", choices=["list", "promote", "champion"])
    reg.add_argument("--config", default="configs/isic_unet.yaml")
    reg.add_argument("--run-id", default="")
    reg.set_defaults(func=_cmd_registry)

    serve = sub.add_parser("serve", help="start the FastAPI + web console")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Model zoo and checkpoint round-trips."""

from __future__ import annotations

import torch

from medmlops.checkpoint import build_model_from_meta, load_checkpoint, new_meta, save_checkpoint
from medmlops.models import UNet, build_densenet, build_model, count_parameters, freeze_backbone


def test_unet_forward_shape_odd_input():
    model = UNet(base_channels=4, depth=2, dropout=0.0).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 65, 47))
    assert out.shape == (2, 1, 65, 47)


def test_unet_parameter_count_and_gradients():
    model = UNet(base_channels=8, depth=2)
    assert count_parameters(model) > 1000
    logits = model(torch.randn(1, 3, 64, 64))
    logits.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_densenet_head_and_freezing():
    model = build_densenet(num_classes=2, pretrained=False)
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 2)
    freeze_backbone(model)
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(not name.startswith("classifier") for name in frozen)


def test_build_model_from_config():
    from medmlops.config import load_config

    cfg = load_config("configs/isic_unet.yaml")
    model = build_model(cfg)
    assert isinstance(model, UNet)
    assert count_parameters(model) > 1_000_000


def test_checkpoint_round_trip(tmp_path):
    model = UNet(base_channels=4, depth=2)
    meta = new_meta(
        task="segmentation",
        arch="unet",
        model_kwargs={"base_channels": 4, "depth": 2},
        image_size=64,
        dataset="unit-test",
        run_id="run-123",
        git_commit="deadbeef",
        config_path="configs/test.yaml",
        threshold=0.45,
    )
    path = save_checkpoint(tmp_path / "tiny.pt", model, meta)
    assert path.exists()
    assert path.with_suffix(".json").exists()

    loaded, loaded_meta = load_checkpoint(path)
    assert loaded_meta["run_id"] == "run-123"
    assert loaded.training is False
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.allclose(a, b)


def test_build_model_rejects_unknown_arch():
    try:
        build_model_from_meta({"arch": "resnet"})
    except ValueError as exc:
        assert "resnet" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

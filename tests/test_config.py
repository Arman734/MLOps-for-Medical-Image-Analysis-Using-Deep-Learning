"""Config loader: nested access, dot paths, env overrides and flattening."""

from __future__ import annotations

import textwrap

import pytest

from medmlops.config import Config, Env, flatten, load_config


@pytest.fixture()
def cfg_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        textwrap.dedent(
            """
            experiment: demo
            data:
              name: isic2018
              image_size: 128
            training:
              epochs: 3
              amp: true
              scales: [0.5, 1.0]
            _private: hidden
            """
        ),
        encoding="utf-8",
    )
    return path


def test_attribute_and_dot_access(cfg_file):
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.experiment == "demo"
    assert cfg.data.image_size == 128
    assert cfg.get_path("training.epochs") == 3
    assert cfg.get_path("training.missing", default=42) == 42


def test_unknown_attribute_raises(cfg_file):
    cfg = load_config(cfg_file)
    with pytest.raises(AttributeError):
        _ = cfg.not_a_key


def test_flatten_drops_private_keys(cfg_file):
    flat = flatten(load_config(cfg_file))
    assert flat["data.image_size"] == 128
    assert flat["training.scales"] == "0.5,1.0"
    assert not any(key.startswith("_") for key in flat)


def test_project_configs_load():
    for name in ("isic_unet", "pneumonia_densenet"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.experiment
        assert cfg.data.image_size > 0
        assert cfg.model.arch in {"unet", "densenet121"}
        assert cfg.mlflow.registry_model_name


def test_env_reads_dotenv_without_overriding(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("MEDMLOPS_DEMO_KEY=from-file\nMEDMLOPS_OTHER=value\n", encoding="utf-8")
    env = Env(dotenv)
    assert env.get("MEDMLOPS_DEMO_KEY") == "from-file"
    assert env.get_bool("MEDMLOPS_DEMO_FLAG", default=True) is True
    monkeypatch.setenv("MEDMLOPS_DEMO_KEY", "from-env")
    assert env.get("MEDMLOPS_DEMO_KEY") == "from-env"

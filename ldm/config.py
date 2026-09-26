"""Strict, portable configuration for the four translation experiments."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"t1n_t1c": ("t1n", "t1c"), "t2w_t2f": ("t2w", "t2f")}
DEFAULTS = {
    "task": "t1n_t1c", "seed": 0,
    "model": {"patch_size": 4, "volume_size": [48, 64, 64], "hidden_size": 512, "depth": 10,
              "global_heads": 8, "local_dim": 64, "bottleneck_dim": 96, "mlp_ratio": 4.0,
              "inner_lr": 1.0, "proj_drop": 0.0, "mdsa_mode": "conditioned", "mdsa_window": [2, 4, 4],
              "mdsa_rank": 8, "mdsa_blocks": [4, 5], "mdsa_gate_hidden": 32},
    "flow": {"time_mean": -0.8, "time_std": 0.8, "time_epsilon": 0.05, "noise_scale": 1.0},
    "training": {"epochs": 600, "batch_size": 1, "num_workers": 0, "lr": 3e-5,
                 "weight_decay": 0.0, "warmup_epochs": 2, "min_lr": 0.0,
                 "ema_decay": 0.9999, "eval_every": 30, "save_best_k": 3},
    "sampling": {"solver": "heun", "steps": 100, "seed": 0, "visualizations": 3},
    "evaluation": {"ssim_window": 7, "reference_patch": [8, 8, 8], "boundary_width": 2,
                   "change_percentile": 95.0, "clip_prediction": True},
    "paths": {"h5": None, "cache": None, "ae": "ae_assets/maisi_v1/autoencoder_v1.pt",
              "output": None},
}


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _merge(base, values, prefix=""):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    for key, value in values.items():
        if key not in base:
            raise ValueError(f"Unknown config field: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, prefix + key + ".")
        else:
            base[key] = value


def validate_config(config):
    if config["task"] not in TASKS:
        raise ValueError("Only t1n_t1c and t2w_t2f translation tasks are supported")
    m, t, s, e, f = [config[k] for k in ("model", "training", "sampling", "evaluation", "flow")]
    if m["patch_size"] not in (4, 8) or tuple(m["volume_size"]) != (48, 64, 64):
        raise ValueError("Only latent P4/P8 on the MAISI 48x64x64 grid are supported")
    for name in ("epochs", "batch_size", "eval_every", "save_best_k"):
        if not isinstance(t[name], int) or t[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if t["num_workers"] < 0 or not 0 <= t["warmup_epochs"] < t["epochs"]:
        raise ValueError("Invalid num_workers/warmup_epochs")
    if not 0 <= t["ema_decay"] < 1 or t["lr"] <= 0 or t["min_lr"] < 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer or EMA settings")
    if s["steps"] < 1 or s["solver"] not in ("heun", "euler") or s["visualizations"] < 0:
        raise ValueError("Invalid sampling settings")
    if e["ssim_window"] < 1 or e["ssim_window"] % 2 != 1 or not 0 <= e["change_percentile"] <= 100:
        raise ValueError("Invalid metric settings")
    if len(e["reference_patch"]) != 3 or min(e["reference_patch"]) < 1 or e["boundary_width"] < 0:
        raise ValueError("Invalid reference grid")
    for section in (m, t, s, e, f):
        if any(isinstance(v, float) and not math.isfinite(v) for v in section.values()):
            raise ValueError("Configuration contains non-finite numbers")
    return config


def load_config(path=None, task=None, patch=None):
    config = deepcopy(DEFAULTS)
    if path:
        with resolve_path(path).open(encoding="utf-8") as handle:
            _merge(config, yaml.safe_load(handle) or {})
    if task is not None:
        config["task"] = task
    if patch is not None:
        config["model"]["patch_size"] = patch
    return validate_config(config)


def dataset_path(config, override=None):
    source, target = TASKS[config["task"]]
    return resolve_path(override or config["paths"]["h5"] or f"data/h5/{source}__{target}_3d.h5")


def run_name(config):
    return f"{config['task']}__ldm__latent-p{config['model']['patch_size']}__d1__mdsa-{config['model']['mdsa_mode']}__seed{config['seed']}"


def model_contract(config):
    return {"format": "ldm_latent_d1_v1", "task": config["task"], "model": config["model"],
            "flow": config["flow"], "local_dilation": 1, "local_kernel": [3, 3, 3],
            "post_fusion": "off", "conditioning": "adaln", "output_head": "linear"}

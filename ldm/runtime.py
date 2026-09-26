import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def amp_dtype(name, device):
    if name == "auto":
        name = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.type == "cuda" else "float32"
    if device.type != "cuda" and name != "float32":
        raise ValueError("CPU execution uses float32")
    if name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use float16 or float32")
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_seed(seed, subject_id):
    return (seed + int(hashlib.sha256(subject_id.encode()).hexdigest()[:8], 16)) % (2 ** 63 - 1)


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact resume is not supported")
        torch.cuda.set_rng_state_all(state["cuda"])

import re
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from ..config import TASKS
from ..runtime import sha256

SPLITS = ("train", "val", "test")
MODEL_SHAPE = (160, 256, 256)


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def patient_id(subject):
    match = re.fullmatch(r"(BraTS-[A-Za-z]+-\d+)-\d+", subject)
    return match.group(1) if match else subject


def inspect_h5(path, task, hash_file=True):
    path = Path(path)
    if task not in TASKS:
        raise ValueError("Unknown translation task")
    result = {"task": task, "source_modality": TASKS[task][0], "target_modality": TASKS[task][1],
              "input_range": "0_1", "splits": {}}
    seen = set()
    with h5py.File(path, "r") as handle:
        for key in ("task", "source_modality", "target_modality"):
            if key in handle.attrs and decode(handle.attrs[key]) != result[key]:
                raise ValueError(f"H5 {key} disagrees with requested task")
        for split in SPLITS:
            group = handle[split]
            for key in ("source", "target", "mask", "subject_id"):
                if key not in group:
                    raise ValueError(f"Missing {split}/{key}")
            shape = group["source"].shape
            if len(shape) != 5 or shape[0] < 1 or shape[1] != 1 or any(v < 1 or v > m for v, m in zip(shape[2:], MODEL_SHAPE)):
                raise ValueError(f"Invalid source shape in {split}: {shape}")
            if group["target"].shape != shape or group["mask"].shape != (shape[0], 3, *shape[2:]):
                raise ValueError(f"Unaligned source/target/mask in {split}")
            if group["subject_id"].shape != (shape[0],):
                raise ValueError(f"Incorrect subject_id shape in {split}")
            ids = [decode(v) for v in group["subject_id"][:]]
            if any(not s.strip() for s in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Empty or duplicate subject IDs in {split}")
            patients = set(patient_id(s) for s in ids)
            if seen & patients:
                raise ValueError(f"Patient overlap across splits: {split}")
            seen.update(patients)
            result["splits"][split] = {"ids": ids, "shape": list(shape)}
    if hash_file:
        result["sha256"] = sha256(path)
    return result


def read_volume(group, index):
    arrays = {k: np.asarray(group[k][index], dtype=np.float32) for k in ("source", "target", "mask")}
    for key, value in arrays.items():
        if not np.isfinite(value).all() or value.min() < -1e-6 or value.max() > 1 + 1e-6:
            raise ValueError(f"{group.name}/{key}[{index}] must be finite [0,1]; no implicit normalization is applied")
    arrays["subject_id"] = decode(group["subject_id"][index])
    return arrays


def prepare_volume(array, device):
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Expected [1,D,H,W]")
    d, h, w = tensor.shape[-3:]
    md, mh, mw = MODEL_SHAPE
    if d > md or h > mh or w > mw:
        raise ValueError("Volume exceeds the model canvas")
    image = F.pad(tensor.unsqueeze(0) * 2 - 1, (0, mw-w, 0, mh-h, 0, md-d), value=-1)
    valid = F.pad(torch.ones_like(tensor).unsqueeze(0), (0, mw-w, 0, mh-h, 0, md-d))
    return image, valid


def crop_original(tensor, shape):
    d, h, w = shape
    return tensor[..., :d, :h, :w]

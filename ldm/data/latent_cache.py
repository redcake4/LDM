"""Task-bound latent caches with strict identity and completion checks."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ..autoencoder import AE_SHA256, LATENT_SHAPE, PREPROCESSING
from ..config import resolve_path
from ..runtime import fingerprint
from .h5_dataset import SPLITS, decode

CACHE_FORMAT = "ldm_paired_maisi_mean_v1"
LATENT_KEYS = ("source_latent", "target_latent", "latent_valid_mask")


def cache_contract(data_info):
    return {"format": CACHE_FORMAT, "dataset": data_info, "ae_sha256": AE_SHA256,
            "preprocessing": PREPROCESSING, "posterior": "mean", "latent_shape": list(LATENT_SHAPE),
            "scale_policy": "inverse_population_std_first_train_target"}


def cache_path(config, data_info, override=None):
    return resolve_path(override or config["paths"]["cache"] or
        f"data/latents/maisi_v1/{config['task']}__{fingerprint(cache_contract(data_info))[:16]}.h5")


def validate_cache(path, contract, require_complete=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Latent cache missing: {path}. Run precompute_latents.py for this task first.")
    with h5py.File(path, "r") as handle:
        if json.loads(handle.attrs.get("contract_json", "{}")) != contract:
            raise ValueError("Cache AE/dataset/task/preprocessing contract mismatch; do not reuse an old cache")
        if require_complete and handle.attrs.get("status") != "complete":
            raise ValueError("Cache incomplete; resume precompute_latents.py before training")
        scale = float(handle.attrs["scale_factor"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Cache scale is not finite and positive")
        storage_dtype = str(handle.attrs["storage_dtype"])
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("Unsupported cache storage dtype")
        for split in SPLITS:
            group = handle[split]
            expected = contract["dataset"]["splits"][split]["ids"]
            ids = [decode(v) for v in group["subject_id"][:]]
            if ids != expected:
                raise ValueError(f"Cache ID order/count differs for {split}")
            complete = group["completed"][:]
            if complete.shape != (len(ids),) or not np.isin(complete, [0, 1]).all():
                raise ValueError("Invalid completion flags")
            if require_complete and not complete.all():
                raise ValueError(f"Cache has unfinished subjects in {split}")
            for key in LATENT_KEYS:
                channels = 1 if key == "latent_valid_mask" else 4
                if group[key].shape != (len(ids), channels, *LATENT_SHAPE[1:]) or group[key].dtype != np.dtype(storage_dtype):
                    raise ValueError(f"Cache shape/dtype mismatch: {split}/{key}")
        return {"contract": contract, "scale_factor": scale, "storage_dtype": storage_dtype,
                "encoding_precision": str(handle.attrs["encoding_precision"])}


class CachedLatentDataset(Dataset):
    def __init__(self, path, split, include_target=True):
        self.path, self.split, self.include_target = str(path), split, include_target
        self.handle, self.pid = None, None
        with h5py.File(path, "r") as f:
            self.ids = [decode(v) for v in f[split]["subject_id"][:]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.handle is None or self.pid != os.getpid():
            self.close()
            self.handle, self.pid = h5py.File(self.path, "r"), os.getpid()
        group = self.handle[self.split]
        keys = LATENT_KEYS if self.include_target else ("source_latent", "latent_valid_mask")
        result = {key: torch.from_numpy(np.asarray(group[key][index], dtype=np.float32)) for key in keys}
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise ValueError(f"Non-finite cached values for {self.ids[index]}")
        mask = result["latent_valid_mask"]
        if mask.min() < 0 or mask.max() > 1 or mask.sum() <= 0:
            raise ValueError("Invalid latent valid mask")
        result["subject_id"] = self.ids[index]
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(handle=None, pid=None)
        return state

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle, self.pid = None, None

    def __del__(self):
        self.close()

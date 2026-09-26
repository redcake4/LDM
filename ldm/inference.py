"""Subject-seeded source-only inference and original-shape H5 export."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from tqdm import tqdm
from .data.h5_dataset import read_volume, crop_original
from .data.latent_cache import CachedLatentDataset
from .evaluation.visualize import save_panel
from .runtime import autocast, subject_seed
from .flow import nfe


@torch.no_grad()
def predict_split(flow, ae, h5_path, cache_path, split, config, device, dtype,
                  prediction_path=None, checkpoint_sha256="", source_sha256="", visual_dir=None):
    data = CachedLatentDataset(cache_path, split, include_target=False)
    flow.eval()
    ae.eval()
    started = time.perf_counter()
    maes = []
    output = None
    started_utc = datetime.now(timezone.utc).isoformat()
    if prediction_path:
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if prediction_path.exists():
            raise FileExistsError(f"Refusing to overwrite predictions: {prediction_path}; use a new output directory")
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    try:
        with h5py.File(h5_path, "r") as source:
            if prediction_path:
                output = h5py.File(prediction_path, "x")
                output.attrs.update(status="incomplete", task=config["task"], split=split, prediction_range="0_1",
                    checkpoint_sha256=checkpoint_sha256, source_h5_sha256=source_sha256,
                    solver=config["sampling"]["solver"], steps=config["sampling"]["steps"],
                    nfe=nfe(config["sampling"]["steps"], config["sampling"]["solver"]),
                    seed=config["sampling"]["seed"], seed_policy="base_seed_plus_sha256_subject_id_v1", weights="ema",
                    created_at=started_utc, amp_dtype=str(dtype), torch_version=str(torch.__version__))
                shape = source[split]["target"].shape
                output.create_dataset("subject_id", data=data.ids, dtype=h5py.string_dtype("utf-8"))
                output.create_dataset("prediction", shape=shape, dtype="float32", chunks=(1, *shape[1:]), compression="lzf")
                output.create_dataset("completed", data=np.zeros(len(data), dtype=np.uint8))
            for index in tqdm(range(len(data)), desc=f"Generate {split}"):
                batch = data[index]
                # No target latent or segmentation enters this generation call.
                condition = batch["source_latent"].unsqueeze(0).to(device)
                seed = subject_seed(config["sampling"]["seed"], batch["subject_id"])
                generator = torch.Generator(device=device).manual_seed(seed)
                with autocast(device, dtype):
                    latent = flow.generate(condition, config["sampling"]["steps"], config["sampling"]["solver"], generator)
                image = ae.decode(latent.float())
                sample = read_volume(source[split], index)
                if sample["subject_id"] != batch["subject_id"]:
                    raise ValueError("Source/cache ID alignment changed")
                prediction = ((crop_original(image, sample["target"].shape[-3:])[0].float().cpu().numpy() + 1) * 0.5).clip(0, 1)
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite prediction; no metrics are accepted")
                maes.append(float(np.abs(prediction - sample["target"]).mean()))
                if output is not None:
                    output["prediction"][index] = prediction
                    output.flush()
                    output["completed"][index] = 1
                    output.flush()
                if visual_dir and index < config["sampling"]["visualizations"]:
                    token = hashlib.sha256(batch["subject_id"].encode()).hexdigest()[:12]
                    save_panel(Path(visual_dir) / f"subject_{index:04d}_{token}.png", sample["source"][0],
                               sample["target"][0], prediction[0], sample["mask"])
            if output is not None:
                output.attrs["status"] = "complete"
    finally:
        data.close()
        if output is not None:
            output.close()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"status": "complete", "split": split, "subjects": len(maes), "full_volume_mae": float(np.mean(maes)),
            "full_volume_mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "range": "0_1", "weights": "ema", "seconds": time.perf_counter()-started,
            "nfe_per_volume": nfe(config["sampling"]["steps"], config["sampling"]["solver"]),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}

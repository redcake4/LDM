import argparse
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm
from .autoencoder import FrozenMaisiAE, LATENT_SHAPE
from .config import TASKS, load_config, dataset_path, resolve_path
from .data.h5_dataset import inspect_h5, read_volume, prepare_volume, SPLITS
from .data.latent_cache import cache_contract, cache_path, validate_cache, LATENT_KEYS
from .runtime import device_for


@torch.no_grad()
def precompute(h5_path, output, contract, ae, device, storage_dtype="float16", resume=False, max_subjects=0):
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("Unsupported storage dtype")
    exists = output.exists()
    precision = "cuda_fp16_autocast" if device.type == "cuda" else "cpu_fp32"
    if exists:
        if not resume:
            raise FileExistsError(f"{output} exists; use --resume to validate and continue")
        metadata = validate_cache(output, contract, require_complete=False)
        if metadata["storage_dtype"] != storage_dtype or metadata["encoding_precision"] != precision:
            raise ValueError("Resume requires identical encoding precision and storage dtype")
        with h5py.File(output, "r") as handle:
            if handle.attrs["status"] == "complete":
                validate_cache(output, contract)
                return {"status": "complete", "already_complete": True, "path": str(output)}
        scale = metadata["scale_factor"]
    else:
        with h5py.File(h5_path, "r") as source:
            first = read_volume(source["train"], 0)
        target, _ = prepare_volume(first["target"], device)
        encoded = ae.encode_unscaled(target).float()
        std = float(encoded.std(unbiased=False))
        if not np.isfinite(std) or std <= 1e-8:
            raise ValueError("Degenerate first-train-target latent scale")
        scale = 1.0 / std
        del target, encoded, first
    ae.set_scale_factor(scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with h5py.File(h5_path, "r") as source, h5py.File(output, "r+" if exists else "x") as dest:
        if not exists:
            dest.attrs.update(contract_json=json.dumps(contract, sort_keys=True), status="incomplete",
                              scale_factor=scale, storage_dtype=storage_dtype, encoding_precision=precision)
            for split in SPLITS:
                ids = contract["dataset"]["splits"][split]["ids"]
                group = dest.create_group(split)
                group.create_dataset("subject_id", data=ids, dtype=h5py.string_dtype("utf-8"))
                group.create_dataset("completed", data=np.zeros(len(ids), dtype=np.uint8))
                for key in LATENT_KEYS:
                    channels = 1 if key == "latent_valid_mask" else 4
                    shape = (len(ids), channels, *LATENT_SHAPE[1:])
                    group.create_dataset(key, shape=shape, dtype=storage_dtype,
                                         chunks=(1, *shape[1:]), compression="lzf")
            dest.flush()
        for split in SPLITS:
            group = dest[split]
            for index in tqdm(range(len(group["completed"])), desc=f"Precompute {split}"):
                if group["completed"][index]:
                    continue
                if max_subjects and written >= max_subjects:
                    return {"status": "incomplete", "written": written, "path": str(output)}
                sample = read_volume(source[split], index)
                source_image, valid = prepare_volume(sample["source"], device)
                target_image, _ = prepare_volume(sample["target"], device)
                values = {"source_latent": ae.encode(source_image), "target_latent": ae.encode(target_image),
                          "latent_valid_mask": ae.latent_valid_mask(valid)}
                for key, value in values.items():
                    array = value[0].float().cpu().numpy().astype(storage_dtype)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite cache value: {split}/{index}/{key}")
                    group[key][index] = array
                dest.flush()
                group["completed"][index] = 1
                dest.flush()
                written += 1
                del source_image, target_image, valid, values, sample
        dest.attrs["status"] = "complete"
    validate_cache(output, contract)
    return {"status": "complete", "written": written, "scale_factor": scale, "path": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Precompute task-specific MAISI posterior-mean latents once")
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--h5-path")
    parser.add_argument("--ae-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=0, help="Debug cap; incomplete cache cannot be trained")
    args = parser.parse_args()
    if args.max_subjects < 0:
        parser.error("--max-subjects cannot be negative")
    config = load_config(task=args.task)
    path = dataset_path(config, args.h5_path)
    print(f"Validating and fingerprinting {path}", flush=True)
    info = inspect_h5(path, args.task)
    device = device_for(args.device)
    ae = FrozenMaisiAE(resolve_path(args.ae_checkpoint or config["paths"]["ae"])).to(device).eval()
    result = precompute(path, cache_path(config, info, args.output), cache_contract(info), ae, device,
                        args.storage_dtype, args.resume, args.max_subjects)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

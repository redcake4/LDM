import argparse
from copy import deepcopy
import json
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .autoencoder import FrozenMaisiAE
from .checkpoint import checked_payload
from .config import TASKS, load_config, normalize_config, dataset_path, resolve_path, run_name, model_contract
from .data.h5_dataset import inspect_h5
from .data.latent_cache import cache_path, cache_contract, validate_cache, CachedLatentDataset
from .flow import LDMFlow
from .inference import predict_split
from .model import LDMModel3D
from .runtime import (write_json, atomic_save, device_for, amp_dtype, autocast, seed_all,
                      rng_state, restore_rng)


def learning_rate(config, progress):
    c = config["training"]
    if progress < c["warmup_epochs"]:
        return c["lr"] * progress / c["warmup_epochs"]
    ratio = (progress - c["warmup_epochs"]) / (c["epochs"] - c["warmup_epochs"])
    return c["min_lr"] + (c["lr"] - c["min_lr"]) * 0.5 * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(source, alpha=1-decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def train(config, h5_path, latent_path, ae_path, output, device, dtype, auto_resume=True):
    config = normalize_config(config)
    output = Path(output)
    print(f"Validating and fingerprinting {h5_path}", flush=True)
    info = inspect_h5(h5_path, config["task"])
    latent_path = cache_path(config, info, latent_path)
    metadata = validate_cache(latent_path, cache_contract(info))
    last = output / "checkpoints/last.pt"
    previous = checked_payload(last) if last.exists() and auto_resume else None
    if previous is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty run directory without an enabled last-checkpoint resume; choose a new output")
    if previous:
        if previous["config"] != config or previous["cache_metadata"] != metadata:
            raise ValueError("Resume requires identical resolved configuration and cache metadata; use a new run for changes")
        if previous["amp_dtype"] != str(dtype) or previous["device_type"] != device.type:
            raise ValueError("Resume precision/device type changed; refusing an unrecorded training change")
    seed_all(config["seed"])
    model = LDMFlow(LDMModel3D(**config["model"]), **config["flow"]).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    decay_params, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay_params).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay_params, "weight_decay": config["training"]["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=config["training"]["lr"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    start, best_records, history = 0, [], []
    if previous:
        model.load_state_dict(previous["model"], strict=True)
        ema.load_state_dict(previous["ema"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        start, best_records, history = previous["next_epoch"], previous["best_records"], previous["history"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_manifest.json", {"model_contract": model_contract(config), "cache_metadata": metadata,
        "h5_path": str(h5_path), "cache_path": str(latent_path), "ae_path": str(ae_path),
        "torch": str(torch.__version__), "amp_dtype": str(dtype), "device": str(device),
        "selection_metric": "full original-volume MAE in [0,1], full val split, EMA weights",
        "sampling_seed_policy": "base_seed_plus_sha256_subject_id_v1", "test_used_for_selection": False})
    if start >= config["training"]["epochs"]:
        print("Requested epochs already completed. Use export_predictions.py for testing.")
        return
    # The frozen AE is separate from model/EMA/optimizer/checkpoint tensors.
    ae = FrozenMaisiAE(ae_path, metadata["scale_factor"]).to(device).eval()
    # AE construction initializes tensors before loading weights. Restore the
    # training RNG afterwards so that resume does not consume a new noise stream.
    if previous:
        restore_rng(previous["rng"])
    dataset = CachedLatentDataset(latent_path, "train")
    started = time.perf_counter()
    try:
        for epoch in range(start, config["training"]["epochs"]):
            model.train()
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                num_workers=config["training"]["num_workers"], pin_memory=device.type == "cuda",
                generator=generator, drop_last=False)
            loss_sum, count = 0.0, 0
            epoch_start = time.perf_counter()
            progress = tqdm(loader, desc=f"LDM train {epoch+1}/{config['training']['epochs']}")
            for index, batch in enumerate(progress):
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(config, epoch + index / len(loader))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                source, target, mask = [batch[key].to(device, non_blocking=True)
                                       for key in ("source_latent", "target_latent", "latent_valid_mask")]
                with autocast(device, dtype):
                    loss = model(source, target, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                old_scale = scaler.get_scale()
                if dtype != torch.float16:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    update_ema(ema, model, config["training"]["ema_decay"])
                loss_sum += float(loss.detach()) * source.shape[0]
                count += source.shape[0]
                progress.set_postfix(loss=f"{float(loss.detach()):.6f}", lr=f"{lr:.2e}")
            record = {"epoch": epoch+1, "train_loss": loss_sum/count, "lr": lr,
                      "train_seconds": time.perf_counter()-epoch_start}
            best_name = None
            expired = []
            if (epoch+1) % config["training"]["eval_every"] == 0 or epoch+1 == config["training"]["epochs"]:
                validation = predict_split(ema, ae, h5_path, latent_path, "val", config, device, dtype,
                    visual_dir=output / f"validation/epoch_{epoch+1:04d}")
                record["validation"] = validation
                metric = validation["full_volume_mae"]
                if not math.isfinite(metric):
                    raise FloatingPointError("Non-finite validation score")
                candidate = {"epoch": epoch+1, "metric": metric, "file": f"best-epoch{epoch+1:04d}-mae{metric:.6f}.pt"}
                ranked = sorted(best_records + [candidate], key=lambda item: (item["metric"], item["epoch"]))
                kept = ranked[:config["training"]["save_best_k"]]
                expired = [item["file"] for item in best_records if item not in kept]
                if candidate in kept:
                    best_name = candidate["file"]
                best_records = kept
            history.append(record)
            payload = {"format": "ldm_training_v1", "config": config, "model_contract": model_contract(config),
                "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "next_epoch": epoch+1, "rng": rng_state(),
                "best_records": best_records, "history": history, "cache_metadata": metadata,
                "amp_dtype": str(dtype), "device_type": device.type}
            if best_name:
                atomic_save(output / "checkpoints" / best_name, payload)
            atomic_save(last, payload)
            write_json(output / "best_checkpoints.json", best_records)
            write_json(output / "training_history.json", history)
            for name in expired:
                if Path(name).name != name or not name.startswith("best-epoch") or not name.endswith(".pt"):
                    raise ValueError("Invalid managed checkpoint filename")
                (output / "checkpoints" / name).unlink(missing_ok=True)
            print(json.dumps(record), flush=True)
        write_json(output / "training_summary.json", {"status": "complete", "epochs_completed": config["training"]["epochs"],
            "best_checkpoints": best_records, "seconds_this_session": time.perf_counter()-started,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description="Train standalone latent LDM, d=1, on paired H5 translation data")
    parser.add_argument("--config")
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--patch", type=int, choices=[2, 4, 8])
    parser.add_argument("--main-structure", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable the structure-based mixer combination; omitted keeps the config setting (default off)")
    parser.add_argument("--h5-path")
    parser.add_argument("--latent-cache")
    parser.add_argument("--ae-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Resolve config/build model only; no data/AE required")
    args = parser.parse_args()
    config = load_config(args.config, args.task, args.patch, main_structure=args.main_structure)
    device = device_for(args.device)
    dtype = amp_dtype(args.amp_dtype, device)
    if args.dry_run:
        model = LDMModel3D(**config["model"])
        print(json.dumps({"config": config, "model_contract": model_contract(config), "h5_path": str(dataset_path(config, args.h5_path)),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "tokens": model.x_embedder.num_patches, "amp_dtype": str(dtype)}, indent=2))
        return
    train(config, dataset_path(config, args.h5_path), args.latent_cache,
          resolve_path(args.ae_checkpoint or config["paths"]["ae"]),
          resolve_path(args.output_dir or config["paths"]["output"] or f"outputs/{run_name(config)}"),
          device, dtype, args.auto_resume)


if __name__ == "__main__":
    main()

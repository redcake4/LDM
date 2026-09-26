"""Real-AE synthetic pipeline smoke test. Outputs are diagnostics, not research results."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from ldm.autoencoder import FrozenMaisiAE
from ldm.config import load_config, resolve_path
from ldm.data.h5_dataset import inspect_h5
from ldm.data.latent_cache import cache_contract, validate_cache
from ldm.precompute import precompute
from ldm.train import train
from ldm.checkpoint import select_checkpoint, checked_payload
from ldm.model import LDMModel3D
from ldm.flow import LDMFlow
from ldm.inference import predict_split
from ldm.evaluation.metrics import evaluate
from ldm.runtime import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint", default="ae_assets/maisi_v1/autoencoder_v1.pt")
    parser.add_argument("--output-dir", default="artifacts/real_ae_pipeline")
    parser.add_argument("--patch", type=int, choices=[2, 4, 8], default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full-volume smoke test requires CUDA")
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data_path, cache_path = output / "synthetic.h5", output / "cache.h5"
    z, y, x = np.ogrid[-1:1:155j, -1:1:256j, -1:1:256j]
    foreground = z*z + (y/0.8)**2 + (x/0.65)**2 < 0.8
    lesion = (z-0.1)**2 + (y+0.2)**2 + (x-0.1)**2 < 0.03
    with h5py.File(data_path, "x") as f:
        f.attrs.update(task="t1n_t1c", source_modality="t1n", target_modality="t1c", synthetic=True)
        for i, split in enumerate(("train", "val", "test")):
            group = f.create_group(split)
            source = (foreground * (0.35 + 0.05*i)).astype(np.float32)
            target = source + lesion.astype(np.float32) * 0.2
            mask = np.zeros((1, 3, 155, 256, 256), dtype=np.uint8)
            mask[0, 0] = lesion
            group.create_dataset("source", data=source[None, None], compression="lzf")
            group.create_dataset("target", data=target[None, None], compression="lzf")
            group.create_dataset("mask", data=mask, compression="lzf")
            group.create_dataset("subject_id", data=[f"SYNTHETIC-{split}"], dtype=h5py.string_dtype())
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ae_path = resolve_path(args.ae_checkpoint)
    ae = FrozenMaisiAE(ae_path).to(device)
    info = inspect_h5(data_path, "t1n_t1c")
    precompute(data_path, cache_path, cache_contract(info), ae, device)
    ae.cpu()
    del ae
    torch.cuda.empty_cache()
    config = load_config(patch=args.patch)
    config["training"].update(epochs=1, warmup_epochs=0, eval_every=1)
    config["sampling"].update(steps=2, visualizations=1)
    run = output / "run"
    train(config, data_path, cache_path, ae_path, run, device, dtype)
    checkpoint = select_checkpoint(run)
    payload = checked_payload(checkpoint)
    model = LDMFlow(LDMModel3D(**config["model"]), **config["flow"]).to(device)
    model.load_state_dict(payload["ema"])
    ae = FrozenMaisiAE(ae_path, payload["cache_metadata"]["scale_factor"]).to(device)
    prediction_path = output / "predictions_test.h5"
    inference = predict_split(model, ae, data_path, cache_path, "test", config, device, dtype,
        prediction_path, sha256(checkpoint), info["sha256"], output / "panels")
    del model, ae, payload
    torch.cuda.empty_cache()
    metrics = evaluate(data_path, prediction_path, output / "metrics", "t1n_t1c", device=device)
    with h5py.File(prediction_path, "r") as f:
        assert f["prediction"].shape == (1, 1, 155, 256, 256)
    write_json(output / "smoke_report.json", {"status": "passed", "synthetic_data_only": True,
        "real_pinned_ae": True, "model_patch": args.patch, "training_epochs": 1, "sampling_steps": 2,
        "original_prediction_shape": [1, 1, 155, 256, 256], "test_subjects": metrics["subjects"],
        "inference": inference, "note": "No clinical image-quality or convergence claim"})
    print(f"Real-AE synthetic workflow passed: {output}")


if __name__ == "__main__":
    main()

import argparse
from copy import deepcopy
from datetime import datetime
import json
import torch
from .autoencoder import FrozenMaisiAE
from .checkpoint import checked_payload, select_checkpoint
from .config import resolve_path, dataset_path
from .data.h5_dataset import inspect_h5
from .data.latent_cache import cache_path, cache_contract, validate_cache
from .model import LDMModel3D
from .flow import LDMFlow
from .inference import predict_split
from .runtime import device_for, amp_dtype, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description="Export EMA predictions using a full-validation selected checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--h5-path")
    parser.add_argument("--latent-cache")
    parser.add_argument("--ae-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = parser.parse_args()
    run_dir = resolve_path(args.run_dir)
    checkpoint = select_checkpoint(run_dir, args.checkpoint)
    payload = checked_payload(checkpoint)
    config = deepcopy(payload["config"])
    h5_path = dataset_path(config, args.h5_path)
    print(f"Validating and fingerprinting {h5_path}", flush=True)
    info = inspect_h5(h5_path, config["task"])
    latent_path = cache_path(config, info, args.latent_cache)
    metadata = validate_cache(latent_path, cache_contract(info))
    if metadata != payload["cache_metadata"]:
        raise ValueError("Checkpoint/cache mismatch (including AE scale, data, precision and posterior)")
    device = device_for(args.device)
    dtype = amp_dtype(args.amp_dtype, device)
    flow = LDMFlow(LDMModel3D(**config["model"]), **config["flow"]).to(device).eval()
    flow.load_state_dict(payload["ema"], strict=True)
    ae = FrozenMaisiAE(resolve_path(args.ae_checkpoint or config["paths"]["ae"]), metadata["scale_factor"]).to(device).eval()
    output = resolve_path(args.output_dir or f"evaluation/{run_dir.name}/{args.split}_{datetime.now():%Y%m%d_%H%M%S}")
    digest = sha256(checkpoint)
    result = predict_split(flow, ae, h5_path, latent_path, args.split, config, device, dtype,
        output / f"predictions_{args.split}.h5", digest, info["sha256"], output / "visualizations")
    result.update(checkpoint=str(checkpoint), checkpoint_sha256=digest, selection=args.checkpoint,
                  amp_dtype=str(dtype), model_contract=payload["model_contract"], sampling=config["sampling"],
                  source_h5_sha256=info["sha256"], cache_metadata=metadata)
    write_json(output / "inference_summary.json", result)
    print(json.dumps({"status": "complete", "subjects": result["subjects"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

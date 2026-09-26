"""Synthetic full-latent model checks plus optional real MAISI roundtrip."""
import argparse
import gc
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ldm.autoencoder import FrozenMaisiAE
from ldm.config import load_config, resolve_path
from ldm.model import LDMModel3D
from ldm.flow import LDMFlow
from ldm.runtime import write_json, autocast, amp_dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint")
    parser.add_argument("--ae-full-volume", action="store_true", help="May exceed small GPUs; explicitly reported if OOM")
    parser.add_argument("--amp-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--updates", type=int, default=3, help="At least three steps to exercise the initially zero-gated backbone")
    parser.add_argument("--output", default="artifacts/cuda_smoke.json")
    parser.add_argument("--patches", type=int, nargs="+", choices=[2, 4, 8], default=[2, 4, 8],
                        help="Latent patch sizes to check; P2 uses 24,576 tokens")
    args = parser.parse_args()
    if args.updates < 3:
        parser.error("--updates must be at least 3 to exercise the initially zero-gated backbone")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script")
    device = torch.device("cuda")
    dtype = amp_dtype(args.amp_dtype, device)
    report = {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "amp_dtype": str(dtype), "synthetic_data_only": True, "model_checks": []}
    for patch in args.patches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        model = LDMFlow(LDMModel3D(**load_config(patch=patch)["model"])).cuda()
        source = torch.randn(1, 4, 48, 64, 64, device="cuda")
        target = torch.randn_like(source)
        started = time.perf_counter()
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        losses = []
        for _ in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, dtype):
                loss = model(source, target, torch.ones_like(source[:, :1]))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if not finite or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite P{patch} smoke result")
            nonzero_mixers = sum(int(p.grad is not None and bool(p.grad.ne(0).any()))
                for name, p in model.named_parameters() if name.endswith(".mixer.qkv.weight"))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        if nonzero_mixers != len(model.net.blocks):
            raise AssertionError("Smoke test did not reach all initially zero-gated mixers")
        torch.cuda.synchronize()
        report["model_checks"].append({"patch": patch, "loss": float(loss.detach()),
            "finite_gradients": bool(finite), "tokens": model.net.x_embedder.num_patches,
            "updates": args.updates, "losses": losses, "nonzero_mixer_gradients": nonzero_mixers,
            "seconds": time.perf_counter()-started, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()})
        del optimizer, scaler, source, target, loss, model
        gc.collect()
    if args.ae_checkpoint:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ae = FrozenMaisiAE(resolve_path(args.ae_checkpoint)).cuda().eval()
        started = time.perf_counter()
        try:
            with torch.no_grad():
                if args.ae_full_volume:
                    source = torch.zeros(1, 1, 160, 256, 256, device="cuda")
                    latent = ae.encode(source)
                    reconstruction = ae.decode(latent)
                else:
                    source = torch.rand(1, 1, 16, 32, 32, device="cuda")
                    with torch.autocast("cuda", dtype=torch.float16):
                        latent, _ = ae.autoencoder.encode(source)
                        reconstruction = ae.autoencoder.decode_stage_2_outputs(latent)
                assert torch.isfinite(latent).all() and torch.isfinite(reconstruction).all()
                assert reconstruction.shape == source.shape
                torch.cuda.synchronize()
                report["ae_check"] = {"status": "passed", "full_volume": args.ae_full_volume,
                    "input_shape": list(source.shape), "latent_shape": list(latent.shape),
                    "seconds": time.perf_counter()-started, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}
        except torch.cuda.OutOfMemoryError:
            report["ae_check"] = {"status": "out_of_memory", "full_volume": args.ae_full_volume,
                                  "note": "Use a larger GPU; this is not a full-volume AE validation pass"}
    write_json(resolve_path(args.output), report)
    print(report)


if __name__ == "__main__":
    main()

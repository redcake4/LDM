"""Audit executable names without deleting third-party academic attribution."""
import argparse
import ast
import gc
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEGACY = re.compile(r"jit|ttt|lsf|3dmijt|3dmjit", re.IGNORECASE)


def audit_static(root=ROOT):
    issues = []
    files = sorted((root / "ldm").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    files += [root / name for name in ("train.py", "precompute_latents.py", "export_predictions.py", "evaluate.py")]
    for path in files:
        relative = path.relative_to(root).as_posix()
        if LEGACY.search(path.stem):
            issues.append(f"{relative}: legacy executable filename")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                names.extend(arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
            for name in names:
                if LEGACY.search(name):
                    issues.append(f"{relative}:{node.lineno}: legacy executable identifier {name}")
            if isinstance(node, ast.ClassDef) and relative in ("ldm/model.py", "ldm/mixer.py", "ldm/flow.py"):
                if not node.name.startswith("LDM"):
                    issues.append(f"{relative}:{node.lineno}: public model class must start with LDM")
    return {"files_checked": len(files), "issues": sorted(set(issues)),
            "scope": "runtime identifiers, imports, model classes and CLI flags; not comments or attribution"}


def audit_contracts():
    from ldm.config import load_config, run_name, model_contract
    from ldm.model import LDMModel3D
    from ldm.data.latent_cache import CACHE_FORMAT
    records = []
    for task in ("t1n_t1c", "t2w_t2f"):
        for patch in (2, 4, 8):
            config = load_config(f"configs/{task}_p{patch}.yaml")
            model = LDMModel3D(**config["model"])
            keys = list(model.state_dict())
            assert not any(LEGACY.search(key) for key in keys), "Legacy state-dict name"
            assert not any("w3_extra" in key or "dilations" in key for key in keys), "Multiscale state found"
            local_keys = [key for key in keys if key.endswith(".mixer.w3")]
            assert len(local_keys) == config["model"]["depth"], "Expected one local kernel per block"
            assert all(tuple(block.mixer.w3.shape[-3:]) == (3, 3, 3) for block in model.blocks)
            assert model_contract(config)["local_dilation"] == 1
            assert config["model"]["mdsa_mode"] == "conditioned"
            active_mdsa = [i for i, block in enumerate(model.blocks) if block.mdsa is not None]
            assert active_mdsa == [4, 5]
            name = run_name(config)
            assert "__ldm__" in name and not LEGACY.search(name)
            records.append({"task": task, "patch": patch, "run_name": name,
                "model_class": type(model).__name__, "tokens": model.x_embedder.num_patches,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "mdsa_blocks_zero_based": active_mdsa, "single_scale_local_kernels": len(local_keys),
                "state_keys_checked": len(keys), "cache_format": CACHE_FORMAT})
            del model
            gc.collect()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output", help="Optional JSON report; relative to the package root")
    args = parser.parse_args()
    report = audit_static()
    report["contracts"] = [] if args.static_only else audit_contracts()
    report["status"] = "failed" if report["issues"] else "passed"
    if args.output:
        from ldm.config import resolve_path
        from ldm.runtime import write_json
        write_json(resolve_path(args.output), report)
    print(json.dumps(report, indent=2))
    return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())

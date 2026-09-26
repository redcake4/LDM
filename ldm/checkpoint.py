from pathlib import Path
from .runtime import load_checkpoint
from .config import model_contract


def select_checkpoint(run_dir, selector="best"):
    import json
    run_dir = Path(run_dir)
    if selector == "best":
        records = json.loads((run_dir / "best_checkpoints.json").read_text(encoding="utf-8"))
        if not records:
            raise ValueError("No full-validation best checkpoint exists")
        name = records[0]["file"]
    elif selector == "last":
        name = "last.pt"
    else:
        raise ValueError("Checkpoint selector must be best or last")
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Invalid checkpoint basename")
    return run_dir / "checkpoints" / name


def checked_payload(path):
    payload = load_checkpoint(path)
    if payload.get("format") != "ldm_training_v1":
        raise ValueError("Not a standalone LDM checkpoint. Legacy/d124/voxel weights cannot be silently loaded")
    if payload["model_contract"] != model_contract(payload["config"]):
        raise ValueError("Checkpoint model contract is inconsistent")
    return payload

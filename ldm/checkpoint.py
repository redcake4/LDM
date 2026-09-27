from pathlib import Path
from .runtime import load_checkpoint
from .config import model_contract, normalize_config


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
    raw_config = payload["config"]
    expected = model_contract(raw_config)
    legacy = "main_structure" not in raw_config["model"]
    if legacy:
        # Only the original schema may omit the flag in both places. Validate
        # that exact contract before adding the default to the loaded payload.
        expected["model"].pop("main_structure")
    else:
        stored_contract = payload["model_contract"]
        stored_model = stored_contract.get("model", {}) if isinstance(stored_contract, dict) else {}
        if not isinstance(stored_model, dict) or type(stored_model.get("main_structure")) is not bool:
            raise ValueError("Checkpoint model contract is inconsistent")
    if payload["model_contract"] != expected:
        raise ValueError("Checkpoint model contract is inconsistent")
    payload["config"] = normalize_config(raw_config)
    payload["model_contract"] = model_contract(payload["config"])
    return payload

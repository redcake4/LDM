import importlib
import json
import sys
from copy import deepcopy
import h5py
import pytest
import torch
from ldm.config import DEFAULTS
from ldm.data.h5_dataset import inspect_h5
from ldm.data.latent_cache import cache_contract
from ldm.precompute import precompute
from ldm.checkpoint import checked_payload, select_checkpoint
from ldm.evaluation.metrics import evaluate
from conftest import FakeAE


@pytest.mark.parametrize("task,patch", [("t1n_t1c", 4), ("t1n_t1c", 8), ("t2w_t2f", 4), ("t2w_t2f", 8)])
def test_train_resume_export_evaluate(paired_h5, tmp_path, monkeypatch, task, patch):
    module = importlib.import_module("ldm.train")
    exporter = importlib.import_module("ldm.export")
    monkeypatch.setattr(module, "FrozenMaisiAE", FakeAE)
    monkeypatch.setattr(exporter, "FrozenMaisiAE", FakeAE)
    with h5py.File(paired_h5, "r+") as f:
        f.attrs.update(task=task, source_modality=task.split("_")[0], target_modality=task.split("_")[1])
    config = deepcopy(DEFAULTS)
    config["task"] = task
    config["model"].update(patch_size=patch, hidden_size=24, depth=2, global_heads=4,
                            local_dim=6, bottleneck_dim=8, mdsa_blocks=[0, 1], mdsa_rank=4)
    config["training"].update(epochs=2, warmup_epochs=0, eval_every=1)
    config["sampling"].update(steps=2, visualizations=0)
    info = inspect_h5(paired_h5, task)
    cache = tmp_path / "cache.h5"
    precompute(paired_h5, cache, cache_contract(info), FakeAE(), torch.device("cpu"))
    output = tmp_path / "run"
    original_predict = module.predict_split

    def fail_on_second_validation(*args, **kwargs):
        if (output / "checkpoints/last.pt").exists():
            raise RuntimeError("simulated interruption after epoch one")
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(module, "predict_split", fail_on_second_validation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.train(config, paired_h5, cache, "unused", output, torch.device("cpu"), torch.float32)
    first = checked_payload(output / "checkpoints/last.pt")
    assert first["next_epoch"] == 1
    monkeypatch.setattr(module, "predict_split", original_predict)
    module.train(config, paired_h5, cache, "unused", output, torch.device("cpu"), torch.float32)
    completed = checked_payload(output / "checkpoints/last.pt")
    assert completed["next_epoch"] == 2
    assert len(completed["history"]) == 2
    assert all("autoencoder" not in key for key in completed["model"])
    assert select_checkpoint(output).is_file()
    if task == "t1n_t1c" and patch == 8:
        uninterrupted = tmp_path / "uninterrupted"
        module.train(config, paired_h5, cache, "unused", uninterrupted, torch.device("cpu"), torch.float32)
        direct = checked_payload(uninterrupted / "checkpoints/last.pt")
        for name, value in completed["model"].items():
            torch.testing.assert_close(value, direct["model"][name], atol=0, rtol=0)
    export = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", ["export_predictions.py", "--run-dir", str(output),
        "--h5-path", str(paired_h5), "--latent-cache", str(cache), "--output-dir", str(export), "--device", "cpu"])
    exporter.main()
    prediction_path = export / "predictions_test.h5"
    summary = evaluate(paired_h5, prediction_path, export / "metrics", task)
    assert summary["subjects"] == 1
    with h5py.File(prediction_path, "r") as f:
        assert f["prediction"].shape == (1, 1, 8, 8, 8)
        assert f.attrs["status"] == "complete"
    # A change of held-out targets/masks does not enter generation. Compare
    # direct inference with the same condition cache and per-subject noise.
    with h5py.File(paired_h5, "r+") as f:
        f["test/target"][:] = 0.1
        f["test/mask"][:] = 0
    from ldm.flow import LDMFlow
    from ldm.model import LDMModel3D
    from ldm.inference import predict_split
    best = checked_payload(select_checkpoint(output))
    model = LDMFlow(LDMModel3D(**config["model"]), **config["flow"])
    model.load_state_dict(best["ema"])
    second_path = tmp_path / "target_changed.h5"
    predict_split(model, FakeAE(scale_factor=best["cache_metadata"]["scale_factor"]), paired_h5, cache,
                  "test", config, torch.device("cpu"), torch.float32, prediction_path=second_path)
    with h5py.File(second_path, "r") as a, h5py.File(prediction_path, "r") as b:
        import numpy as np
        np.testing.assert_array_equal(a["prediction"][:], b["prediction"][:])

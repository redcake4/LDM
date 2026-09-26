from copy import deepcopy
import json
import h5py
import numpy as np
import pytest
import torch
from ldm.config import DEFAULTS, load_config, resolve_path, ROOT
from ldm.data.h5_dataset import inspect_h5, prepare_volume, crop_original
from ldm.data.latent_cache import cache_contract, validate_cache, CachedLatentDataset
from ldm.precompute import precompute
from ldm.evaluation.metrics import score_subject, evaluate
from ldm.autoencoder import FrozenMaisiAE
from conftest import FakeAE


def test_original_depth_roundtrip():
    x = np.zeros((1, 155, 256, 256), dtype=np.float32)
    prepared, valid = prepare_volume(x, "cpu")
    assert prepared.shape == (1, 1, 160, 256, 256)
    assert valid[..., 155:, :, :].sum() == 0
    padded = FrozenMaisiAE.pad(prepared)
    assert padded.shape[-3:] == (192, 256, 256)
    restored = crop_original(padded[..., 16:176, :, :], x.shape[-3:])
    torch.testing.assert_close(restored, torch.full((1, 1, 155, 256, 256), -1.0))


def test_split_patient_overlap_rejected(paired_h5):
    with h5py.File(paired_h5, "r+") as f:
        f["train/subject_id"][0] = "BraTS-GLI-00001-000"
        f["val/subject_id"][0] = "BraTS-GLI-00001-001"
    with pytest.raises(ValueError, match="overlap"):
        inspect_h5(paired_h5, "t1n_t1c")


def test_cache_resume_and_identity(paired_h5, tmp_path):
    contract = cache_contract(inspect_h5(paired_h5, "t1n_t1c"))
    path = tmp_path / "latents.h5"
    result = precompute(paired_h5, path, contract, FakeAE(), torch.device("cpu"), max_subjects=1)
    assert result["status"] == "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        validate_cache(path, contract)
    result = precompute(paired_h5, path, contract, FakeAE(), torch.device("cpu"), resume=True)
    assert result["status"] == "complete"
    meta = validate_cache(path, contract)
    assert meta["scale_factor"] > 0
    data = CachedLatentDataset(path, "test", include_target=False)
    assert "target_latent" not in data[0]
    assert data[0]["source_latent"].shape == (4, 48, 64, 64)
    data.close()
    bad = deepcopy(contract)
    bad["posterior"] = "sample"
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_cache(path, bad)


def test_metric_regions_and_range():
    source = np.zeros((8, 8, 8), np.float32)
    source[1:7, 1:7, 1:7] = 0.5
    mask = np.zeros((3, 8, 8, 8), np.float32)
    mask[0, 3:5, 3:5, 3:5] = 1
    exact = score_subject(source, source, source, mask, DEFAULTS["evaluation"])
    assert exact["full_volume_mae"] == 0
    assert exact["brain_psnr_db"] == 120
    assert abs(exact["full_volume_ssim3d"] - 1) < 1e-6
    shifted = score_subject(source + 0.1, source, source, mask, DEFAULTS["evaluation"])
    assert shifted["full_volume_mae"] == pytest.approx(0.1, abs=1e-6)
    assert shifted["full_volume_psnr_db"] == pytest.approx(20, abs=1e-4)


def test_strict_prediction_alignment(paired_h5, tmp_path):
    output = tmp_path / "prediction.h5"
    with h5py.File(paired_h5, "r") as source, h5py.File(output, "w") as f:
        f.attrs.update(status="complete", prediction_range="0_1", split="test", task="t1n_t1c")
        f.create_dataset("subject_id", data=source["test/subject_id"][:])
        f.create_dataset("prediction", data=source["test/target"][:])
    summary = evaluate(paired_h5, output, tmp_path / "metrics", "t1n_t1c")
    assert summary["metrics"]["full_volume_mae"]["mean"] == 0
    with h5py.File(output, "r+") as f:
        f["subject_id"][0] = "wrong-subject"
    with pytest.raises(ValueError, match="IDs"):
        evaluate(paired_h5, output, tmp_path / "metrics_bad", "t1n_t1c")


def test_paths_anchor_and_unknown_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/h5/test.h5") == ROOT / "data/h5/test.h5"
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("model:\n  local_dilations: [1, 2, 4]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config"):
        load_config(cfg)


@pytest.mark.parametrize("markers", [[0], [2], [[1]]])
def test_incomplete_prediction_markers(paired_h5, tmp_path, markers):
    output = tmp_path / "incomplete.h5"
    with h5py.File(paired_h5, "r") as source, h5py.File(output, "w") as f:
        f.attrs.update(status="complete", prediction_range="0_1")
        f.create_dataset("subject_id", data=source["test/subject_id"][:])
        f.create_dataset("prediction", data=source["test/target"][:])
        f.create_dataset("completed", data=markers)
    with pytest.raises(ValueError, match="completion markers"):
        evaluate(paired_h5, output, tmp_path / "metrics_incomplete", "t1n_t1c")

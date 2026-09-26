import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from .metric_core import uniform_ssim_map, patch_boundary_mask, region_metrics, add_region, summarize
from ..config import TASKS, DEFAULTS, resolve_path, load_config, dataset_path
from ..data.h5_dataset import inspect_h5, read_volume, decode
from ..runtime import write_json, device_for


def score_subject(prediction, source, target, segmentation, settings, device="cpu"):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction must have exactly the original target shape, without padding")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    record = {"prediction_out_of_range_fraction": float(((prediction < 0) | (prediction > 1)).mean())}
    if settings["clip_prediction"]:
        prediction = np.clip(prediction, 0, 1)
    if min(target.shape) < settings["ssim_window"]:
        raise ValueError("Volume is smaller than the SSIM window")
    ssim = uniform_ssim_map(prediction, target, settings["ssim_window"], str(device))
    brain = (np.abs(source) > 1e-6) | (np.abs(target) > 1e-6)
    boundary = patch_boundary_mask(target.shape, settings["reference_patch"], settings["boundary_width"])
    change = np.abs(target - source)
    threshold = float(np.percentile(change[brain], settings["change_percentile"])) if brain.any() else float("inf")
    regions = {"full_volume": np.ones(target.shape, bool), "brain": brain, "background": ~brain,
               "patch_boundary": boundary, "patch_interior": ~boundary,
               "change_roi": brain & (change >= threshold), "seg_union": (segmentation > 0.5).any(0)}
    for channel in range(segmentation.shape[0]):
        regions[f"seg_{channel+1}"] = segmentation[channel] > 0.5
    for name, mask in regions.items():
        add_region(record, name, region_metrics(prediction, target, ssim, mask))
    record["change_threshold"] = threshold
    return record


def evaluate(h5_path, predictions, output_dir, task, split="test", prediction_range=None, settings=None, device="cpu"):
    settings = dict(settings or DEFAULTS["evaluation"])
    output_dir = Path(output_dir)
    info = inspect_h5(h5_path, task)
    rows = []
    with h5py.File(h5_path, "r") as source, h5py.File(predictions, "r") as pred:
        if "status" in pred.attrs and pred.attrs["status"] != "complete":
            raise ValueError("Prediction file is incomplete")
        if "task" in pred.attrs and pred.attrs["task"] != task:
            raise ValueError("Prediction task mismatch")
        if "split" in pred.attrs and pred.attrs["split"] != split:
            raise ValueError("Prediction split mismatch")
        if "source_h5_sha256" in pred.attrs and pred.attrs["source_h5_sha256"] != info["sha256"]:
            raise ValueError("Prediction dataset fingerprint mismatch")
        group = pred[split] if split in pred else pred
        if "subject_id" not in group or "prediction" not in group:
            raise ValueError("Prediction H5 must contain subject_id and prediction")
        ids = [decode(v) for v in group["subject_id"][:]]
        expected = info["splits"][split]["ids"]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError("Prediction IDs must cover the entire split exactly, without duplicates/extras")
        if "completed" in group:
            completed = group["completed"]
            if completed.shape != (len(ids),) or not np.all(completed[:] == 1):
                raise ValueError("Prediction completion markers are incomplete or malformed")
        if group["prediction"].shape != (len(ids), *source[split]["target"].shape[1:]):
            raise ValueError("Prediction dimensions differ from original H5; regenerate or explicitly restore geometry")
        declared_range = pred.attrs.get("prediction_range")
        if prediction_range and declared_range and prediction_range != declared_range:
            raise ValueError("Requested intensity range contradicts prediction metadata")
        value_range = prediction_range or declared_range
        if value_range not in {"0_1", "minus1_1"}:
            raise ValueError("Specify prediction_range; it is never guessed from intensity values")
        lookup = {subject: i for i, subject in enumerate(ids)}
        for index, subject in enumerate(tqdm(expected, desc=f"Evaluate {split}")):
            sample = read_volume(source[split], index)
            volume = np.asarray(group["prediction"][lookup[subject], 0], dtype=np.float32)
            if value_range == "minus1_1":
                volume = (volume + 1) * 0.5
            row = score_subject(volume, sample["source"][0], sample["target"][0], sample["mask"], settings, device)
            rows.append({"subject_id": subject, "index": index, **row})
        checkpoint_hash = str(pred.attrs.get("checkpoint_sha256", "unrecorded_external_prediction"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_per_subject.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows)
    report.update(task=task, split=split, checkpoint_sha256=checkpoint_hash, source_h5_sha256=info["sha256"],
                  protocol={**settings, "metric_range": "0_1", "data_range": 1.0,
                    "original_volume_only": True, "background_zeroing": False,
                    "brain_mask": "abs(source)>1e-6 OR abs(target)>1e-6",
                    "ssim": "3D uniform window, zero padding, include pad in averaging; region average of map",
                    "aggregation": "subject-wise mean and sample SD (ddof=1), finite values only",
                    "empty_region": "NaN in CSV; null in JSON; excluded from mean/count"})
    write_json(output_dir / "metrics_summary.json", report)
    print(f"Saved unified metrics to {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete ID-aligned H5 predictions without re-running the model")
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--h5-path")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--prediction-range", choices=["0_1", "minus1_1"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    evaluate(dataset_path(load_config(task=args.task), args.h5_path), resolve_path(args.predictions),
             resolve_path(args.output_dir), args.task, args.split, args.prediction_range, device=device_for(args.device))


if __name__ == "__main__":
    main()

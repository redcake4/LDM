# Derived from the recorded parent implementation; see README.md#acknowledgements.
import math
import numpy as np

def uniform_ssim_map(prediction, target, window=7, device="cpu"):
    import torch
    import torch.nn.functional as functional

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(prediction).float()[None, None].to(device)
    y = torch.from_numpy(target).float()[None, None].to(device)
    padding = window // 2
    mu_x = functional.avg_pool3d(x, window, stride=1, padding=padding)
    mu_y = functional.avg_pool3d(y, window, stride=1, padding=padding)
    sigma_x = functional.avg_pool3d(x * x, window, stride=1, padding=padding) - mu_x.square()
    sigma_y = functional.avg_pool3d(y * y, window, stride=1, padding=padding) - mu_y.square()
    sigma_xy = functional.avg_pool3d(x * y, window, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    result = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return result[0, 0].cpu().numpy()


def patch_boundary_mask(shape, patch_size, width):
    mask = np.zeros(shape, dtype=bool)
    for axis, patch in enumerate(patch_size):
        for boundary in range(int(patch), shape[axis], int(patch)):
            start = max(0, boundary - width)
            stop = min(shape[axis], boundary + width)
            slices = [slice(None)] * 3
            slices[axis] = slice(start, stop)
            mask[tuple(slices)] = True
    return mask


def region_metrics(prediction, target, ssim_map, mask):
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return {"mae": math.nan, "mse": math.nan, "psnr_db": math.nan, "ssim3d": math.nan, "voxels": 0}
    error = prediction[mask] - target[mask]
    mse = float(np.mean(error * error))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "psnr_db": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
        "ssim3d": float(np.mean(ssim_map[mask])),
        "voxels": count,
    }


def add_region(record, name, values):
    for metric, value in values.items():
        record[f"{name}_{metric}"] = value


def summarize(records):
    summary = {"subjects": len(records), "metrics": {}}
    keys = sorted({key for row in records for key, value in row.items() if isinstance(value, (int, float))})
    for key in keys:
        values = np.asarray([row.get(key, math.nan) for row in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary["metrics"][key] = {
            "mean": float(finite.mean()) if finite.size else math.nan,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "count": int(finite.size),
        }
    return summary

"""Read-only H5 contract and optional intensity scan."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from tqdm import tqdm
from ldm.config import TASKS, load_config, dataset_path
from ldm.data.h5_dataset import inspect_h5, read_volume, SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--h5-path")
    parser.add_argument("--scan", action="store_true", help="Validate finite [0,1] intensities for every volume")
    args = parser.parse_args()
    path = dataset_path(load_config(task=args.task), args.h5_path)
    info = inspect_h5(path, args.task)
    if args.scan:
        with h5py.File(path, "r") as handle:
            for split in SPLITS:
                for index in tqdm(range(len(handle[split]["subject_id"])), desc=f"Check {split}"):
                    read_volume(handle[split], index)
    print(json.dumps({"path": str(path), "sha256": info["sha256"], "task": args.task,
        "splits": {s: {"subjects": len(v["ids"]), "shape": v["shape"]} for s, v in info["splits"].items()},
        "all_values_scanned": args.scan}, indent=2))


if __name__ == "__main__":
    main()

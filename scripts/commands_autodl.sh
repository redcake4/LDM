#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TASK="${TASK:-t1n_t1c}"
PATCH="${PATCH:-4}"
python scripts/check_dataset.py --task "$TASK" --scan
python scripts/download_ae.py
python -u precompute_latents.py --task "$TASK" --device cuda --resume
python -u train.py --config "configs/${TASK}_p${PATCH}.yaml" --device cuda --amp-dtype auto

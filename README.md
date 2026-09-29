# Latent Space Is Not Flat: Rethinking Latent Structure for 3D Medical Image Synthesis

## Setup

Use Python 3.10+ and CUDA-enabled PyTorch 2.5+. Run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Data Preparation

Place the paired H5 datasets at:

- T1N-to-T1C: `data/h5/t1n__t1c_3d.h5`
- T2W-to-T2F: `data/h5/t2w__t2f_3d.h5`

Each file needs nonempty, patient-disjoint `train`, `val`, and `test` groups containing `source` and `target`
with shape `[N,1,D,H,W]`, `mask` with shape `[N,3,D,H,W]`, and `subject_id` with shape `[N]`.
Use unique, nonempty subject IDs and image/mask values within `[0,1]` (all finite).
The existing data uses volumes of size `155x256x256`.

```bash
python scripts/check_dataset.py --task t1n_t1c --scan
python scripts/download_ae.py
python -u precompute_latents.py --task t1n_t1c --device cuda --resume
```

The download script saves the autoencoder to `ae_assets/maisi_v1/autoencoder_v1.pt`.
For T2 data, use `--task t2w_t2f`. Complete precomputation before training.

## Train

```bash
python -u train.py --config configs/t1n_t1c_p4.yaml --output-dir outputs/t1_p4 --device cuda
```

Choose a T1/T2 P2/P4/P8 configuration from [configs](configs).
Add `--main-structure` to enable the optional structural shortcut; it is off by default.
Repeating the same command resumes training. Use a new `--output-dir` when changing settings.

## Evaluate

After training:

```bash
python -u export_predictions.py --run-dir outputs/t1_p4 --checkpoint best --split test --output-dir evaluation/t1_p4_test --device cuda
python -u evaluate.py --task t1n_t1c --predictions evaluation/t1_p4_test/predictions_test.h5 --split test --output-dir evaluation/t1_p4_test/metrics --device cuda
```

Use the matching run directory and task for other configurations. Export restores the model settings
from the checkpoint. Choose a new export directory when predictions already exist.
For a dataset outside the default path, pass the same `--h5-path` to checking, precomputation,
training, export, and evaluation.

## Acknowledgements

[BraTS](https://www.synapse.org/Synapse:syn51156910),
[NVIDIA MAISI](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR),
and [MONAI](https://github.com/Project-MONAI/MONAI).

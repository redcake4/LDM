# Manuscript-to-code alignment

Paper: *Latent Space Is Not Flat: Rethinking Latent Structure for 3D Medical Image Synthesis*
([arXiv:2609.32794v1](https://arxiv.org/abs/2609.32794), 26 Sep 2026).

This document states exactly which parts of the manuscript are implemented in this repository,
which are approximations, and which are not present at all. It exists so that a reader can tell,
without reading the source, what the published code does and does not support.

**Status summary.** The repository provides a tested training/export/evaluation pipeline for the two
cross-modality translation tasks, but it is **not** the configuration reported in the paper. Several
core elements of Latent Structure Flow (LSF) are either missing or set to different values, and the
latent-space analysis and inpainting experiments are absent. Treat the code as a partial generator
implementation, not as a reproduction package.

Legend: **Match** = faithful. **Approximate** = same idea, different detail. **Missing** = not in this repository.

## Method

| Manuscript item | Code location | Status | Notes |
| --- | --- | --- | --- |
| Eq. (1) windowed low-rank split `S=PX`, `R=(I-P)X` | `ldm/mdsa.py` `MDSA3D.decompose` | Match | Per-window Gram, top-`r` eigenvectors via `torch.linalg.eigh`. |
| Eq. (3) basis estimated **jointly from state and condition**, `U_w = eig_r(Γ(C_w) + Γ(H_w))` with `Γ(A) = AAᵀ/(tr(AAᵀ)+η)` | `ldm/mdsa.py` `_normalized_gram`, `decompose`; `ldm/model.py` `c_embedder` | Match | Enabled with `model.mdsa_basis: joint`. `state` (the previous behaviour) and `condition` are also available, which covers the "state only / condition only / joint" ablation. Default remains `state` for backward compatibility. |
| Eq. (4) routing `S→global`, `R→local`, gated fusion | `ldm/model.py` `LDMBlock3D.forward` → `ldm/mixer.py` | Match | Streams enter via separate slices of the shared `qkv` projection. |
| Eq. (2) inner-update operator, `λ=1` | `ldm/mixer.py` `global_update` / `local_update` | Match | One normalised gradient step from a learned initialisation. |
| Global branch `f(q) = (qΘ₁) ⊙ SiLU(qΘ₂)`, 8 heads | `ldm/mixer.py` `global_update` | Match | |
| Local branch: depthwise 3×3×3, 64 channels, **dilations {1,2,4} averaged** | `ldm/mixer.py` `local_update` | Match | Enabled with `model.local_dilations: [1, 2, 4]`. The inner gradient is taken over the dilated support, and the kernel is shared across dilations, so this adds no parameters. Default remains `[1]`. |
| Gates `g∈(0,2)` from `S`, `‖S‖²`, `‖R‖²` and the time embedding, initialised to 1 | `ldm/mdsa.py` `_gates` | Match | `2·sigmoid(·)` with zero-initialised output layer gives exactly 1.0 at init. |
| Fused output normalised by a time-modulated RMSNorm | `ldm/model.py` `norm_fusion` / `fusion_modulation` | Match | Enabled with `model.post_fusion_norm: true`; adds 5.258 M. Zero-initialised, so it is an identity at initialisation. Default remains off. |
| Eq. (5) output composition `X_out = S + ΔŜ_g + R̂_l`, replacing the full-input shortcut | `ldm/model.py` `LDMBlock3D.forward` | **Approximate** | The code computes `shortcut + gate_mixer · mixed`, where `mixed` is the single fused projection of both branches and `gate_mixer` is an adaLN gate. The paper adds two separately projected predictions to `S` with no adaLN gate on that sum. Also, the projector is estimated from the modulated features `H` but applied to the **raw residual-stream** tensor `X`, whereas the paper sets `X_in,w = H_w`. |
| Projector detached from the gradient | `ldm/model.py` (`detach_projector=True`, hard-coded) | Match | Not configurable, so it cannot be ablated. |
| Flow objective `L = E‖ẑ−z‖²/a_τ²`, `a_τ = max(1−τ, ε)` | `ldm/flow.py` `LDMFlow.forward` | Match | |
| Sampler | `ldm/flow.py` `LDMFlow.generate` | Approximate | Paper uses **Euler**; repository defaults to `heun`. Euler is available. |
| Frozen MAISI codec, unchanged loss | `ldm/autoencoder.py` | Match | |
| Inpainting: mask appended to `z_c` as an extra channel; `Γ(C_w)=0` in fully masked windows; known voxels copied back after decoding | — | **Missing** | Not implemented; see Experiments. |

## Settings

Paper values are from §3 "Implementation". `K = 2·4·4 = 32` tokens per window throughout.

| Setting | Paper | This repository | Status |
| --- | --- | --- | --- |
| Blocks `L`, hidden size | 10, 512 | 10, 512 | Match |
| Global heads | 8 | 8 | Match |
| Local channels | 64 | 64 | Match |
| Inner step `λ` | 1 | 1.0 | Match |
| Patch size `p` | **2** (token grid 24×32×32) | README's worked example uses `p4` | **Differs** |
| Window | 2×4×4 | 2×4×4 | Match |
| **Rank `r`** | **2** | 8 in `*_p*.yaml`, **2** in `paper_lsf_*.yaml` | Fixed |
| **Blocks carrying the split** | **every block** | `[4,5]` in `*_p*.yaml`, **all 10** in `paper_lsf_*.yaml` | Fixed |
| Structural shortcut | Core method | off in `*_p*.yaml`, **on** in `paper_lsf_*.yaml` | Fixed |
| Local dilations | {1, 2, 4} | `[1]` by default, **[1,2,4]** in `paper_lsf_*.yaml` | Fixed |
| Basis source | Joint (Eq. 3) | `state` by default, **`joint`** in `paper_lsf_*.yaml` | Fixed |
| Optimizer | AdamW, β=(0.9, 0.95), no weight decay | Same | Match |
| Learning rate | **constant** 3e-5 after 2 warm-up epochs | 3e-5 with **cosine decay** to `min_lr` | **Differs** |
| Epochs | **80** | **600** | **Differs** |
| Batch size, precision, EMA | 1, bf16, 0.9999 | 1, bf16, 0.9999 | Match |
| Sampling | **Euler, 50 steps** | `heun`, 100 steps | **Differs** |
| Checkpoint selection | Validation MAE inside the hole (inpainting) | Full-volume validation MAE | Differs (task-dependent) |
| Parameters / FLOPs | 62.9 M / 1.72 TFLOPs per forward | **55.84 M trainable, 68.48 M including `pos_embed`** | **Still unexplained** |
| Data | BraTS-GLI, 8:1:1, test `n=125` | Split manifests not published | **Missing** |

## Experiments

| Manuscript item | Status | Notes |
| --- | --- | --- |
| §3.1 latent analysis, Table 1 (3 codecs, 10 volumes, 5 perturbation families) | **Missing** | No perturbation, eigen-energy or decoder-sensitivity code. Requires a KL-VAE and a VQ-VAE in addition to MAISI; neither is referenced anywhere in the repository. |
| Fig. 3 rank selection (`r ∈ {2,4,8}`, decoded component energy) | **Missing** | This is the experiment that justifies `r=2`. |
| Table 2 translation vs cWDM / PMRF / MoTFM / WFM | Partial | Pipeline present; no baselines, no paired statistics, no multi-seed. |
| Table 3 inpainting vs LeFusion / DiffTumor / cWDM, 4 contrasts | **Missing** | `ldm/config.py` `TASKS` defines only `t1n_t1c` and `t2w_t2f`. |
| Table 4 ablation (a)–(f), including the routing swap (e) | **Missing** | `mdsa_mode` has no routing-swap option; rows (a)–(c) are partially reachable via `off`/`gate_only`, row (e) is not reachable at all. |
| `ρ_S` structural fraction, Eq. (6) | **Missing** | Not computed anywhere. |
| LPIPS with MedicalNet ResNet-10 features | **Missing** | `ldm/evaluation/metrics.py` has PSNR/SSIM/MAE over region masks only. |
| Inference cost (FLOPs, latency, peak memory, eigendecomposition overhead) | Partial | `scripts/smoke_cuda.py` records peak CUDA allocation on synthetic data only. |

## Open issue: the Gaussian reference for `ρ_S`

The two Gaussian reference values quoted in the paper are not consistent with each other.

Eq. (6) defines `ρ_S` using `P*_w`, the projector computed **from the target latent `z_t`**. For an error
`δ` that is independent of `z_t` — which is the case for i.i.d. Gaussian `δ` — that projector is fixed
with respect to `δ`, so

```
E[ρ_S] = rank / K = r / 32
```

exactly.

- **Table 4** uses `r = 2` and quotes a Gaussian reference of **6.25 %**. That is exactly `2/32`. Consistent.
- **Table 1(B)** uses `r = 8` (§3.1: "`S` and `R` use rank 8") and reports **33–34 %**. The same formula
  gives `8/32 = 25 %`.

Table 4 confirms the `r/K` formula, so the `r=8` row is the one that needs explaining. The 33–34 % value is also suspiciously stable across
three very different codecs (34 / 34 / 33), which points at a definitional cause rather than codec
behaviour.

### Resolved: `scripts/rho_s_null.py`

`python scripts/rho_s_null.py --ranks 1 2 4 8 16 --trials 3000` measures the null distribution on pure
Gaussian tensors under three conventions for where the basis comes from (`K = 32`):

| `r` | `r/K` | basis from `z` (Eq. 6 as written) | basis from `z+δ` | **basis from `δ` itself** |
| --- | --- | --- | --- | --- |
| 1 | 3.12 % | 3.13 % | 3.14 % | 4.74 % |
| 2 | 6.25 % | **6.25 %** | 6.29 % | 9.26 % |
| 4 | 12.50 % | 12.51 % | 12.56 % | 17.86 % |
| 8 | 25.00 % | 25.00 % | 25.08 % | **33.62 %** |
| 16 | 50.00 % | 49.99 % | 50.10 % | 60.68 % |

This settles it:

- Eq. (6) as written reproduces `r/K` to three significant figures at every rank, confirming
  **Table 4's 6.25 % is correct** for `r = 2`.
- Estimating the basis from the **error itself** at `r = 8` gives **33.62 %**, which is exactly the
  **33–34 %** that Table 1(B) reports, and explains why that value barely moves across three very
  different codecs (34 / 34 / 33).

So Table 1(B) and Table 4 are measuring the structural fraction under **two different conventions**,
and the paper compares them as though they were the same quantity. Either Table 1(B)'s reference
should be recomputed with the projector taken from `z` (giving 25 %), or the text should state that
the two tables use different projector definitions. The earlier Marchenko–Pastur framing in this
document was imprecise: the inflation is real, but it comes from estimating the basis on `δ`, not
from sampling noise in a basis estimated on `z`.

## Trial run

A dependency-only smoke test that needs no data and no GPU:

```bash
python train.py --config configs/paper_lsf_t1n_t1c.yaml --dry-run
```

This resolves the configuration, builds the model and prints the parameter count, the token count and
the resolved contract.

### Measured on `configs/paper_lsf_t1n_t1c.yaml` (CPU, torch 2.4.1)

```
run_name   t1n_t1c__ldm__latent-p2__d1__mdsa-conditioned__seed0__main-structure__basis-joint
trainable  55.840M
frozen     12.636M = sincos pos_embed 12.583M + condition embedder 0.053M
tokens     24576   grid=(24, 32, 32)   <-- matches the paper
r=2, window [2,4,4], split in 10/10 blocks, shortcut in 10, basis=joint
sampling   euler, 50 steps (NFE 50)    <-- matches the paper
lr         [0, 1.5e-5, 3e-5, 3e-5, 3e-5, 3e-5] at epochs [0,1,2,10,40,79]; constant: True
```

Three AdamW steps with every paper option enabled, on the small P8 grid:

```
step 0: loss=5.268249 grad_norm=2.703e-01 tensors_with_grad=2
step 1: loss=1.478813 grad_norm=6.630e-02 tensors_with_grad=28
step 2: loss=1.411864 grad_norm=5.664e-02 tensors_with_grad=212
euler sample: (1, 4, 48, 64, 64) finite=True
```

Backward compatibility is verified: with default settings the state-dict keys, the parameter count and
the output are bit-identical to the previous implementation, and `local_dilations=[1]` reproduces the
old local branch exactly.

> The `tensors_with_grad=2` at step 0 is **expected, not a bug**. `initialize_weights` zero-initialises
> every adaLN gate and the output head, so the network emits exactly zero on the first pass and only
> `final_layer.linear.weight` and `.bias` receive gradient. Gradient reaches the rest of the network
> from step 1 onwards.

### Unresolved: the parameter count

`python scripts/param_audit.py` accounts for the difference against the paper's 62.9 M. With every
paper option now enabled the model has **55.840 M trainable** parameters, or **68.475 M** counting the
non-trainable sincos positional embedding. **Neither reading is 62.9 M**, and the paper does not say
which convention it used.

The audit also prices each remaining structural difference and reports that **no combination of them
lands within 0.5 M of 62.9 M**:

| Candidate | Delta | Resulting total |
| --- | --- | --- |
| SwiGLU width without the `2/3` reduction in `ldm/layers.py` | +10.505 M | 66.344 M (overshoots) |
| Per-branch output projections for Eq. (4)/(5) instead of one shared `Proj` | +0.005 M | 55.845 M |

The released block therefore differs from the block evaluated in the paper in a way that is still not
documented. **The authors should publish the exact block definition and state whether the 62.9 M
figure includes the positional embedding.** Until then no run from this repository can be claimed to
match the paper's model capacity.

### Open question: the condition embedding receives no gradient

The paper states that the basis is detached. Since the condition features of Eq. (3) feed nothing but
the basis, no gradient can ever reach the condition patch embedding, which would make it a fixed random
projection. This implementation therefore freezes `c_embedder` explicitly rather than reporting it as
trainable while it never changes. If the paper intends the condition embedding to be trained — for
example by sharing it with another consumer, or by detaching only the eigendecomposition — that should
be stated, because it changes both the parameter count and the method.

## Reproducibility status

- Tag `v1.0.0` marks the initial public release; `v2.0.0` adds the components and audits above.
- **No run in this repository has reproduced a number printed in the manuscript.** The parameter
  count does not yet match either (see the Settings table and `scripts/param_audit.py`).

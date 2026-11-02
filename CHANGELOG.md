# Changelog

## v2.0.0

Brings the implementation and its documentation in line with
[arXiv:2609.32794v1](https://arxiv.org/abs/2609.32794) and adds two audit tools.

### Added

- `model.mdsa_basis` — `state`, `condition` or `joint`. `joint` implements Eq. (3),
  `U_w = eig_r(Γ(C_w) + Γ(H_w))` with `Γ(A) = AAᵀ/(tr(AAᵀ)+η)`, using a separate condition patch
  embedding. The three values also provide the state-only / condition-only / joint ablation.
- `model.local_dilations` — the depthwise kernel is applied at `{1, 2, 4}` and averaged, with the
  inner gradient taken over the dilated support. The kernel is shared, so this adds no parameters.
- `model.post_fusion_norm` — time-modulated RMSNorm on the fused mixer output. Zero-initialised, so
  it is an identity at initialisation. Adds 5.258 M parameters.
- `configs/paper_lsf_t1n_t1c.yaml` and `configs/paper_lsf_t2w_t2f.yaml` — every setting the paper
  reports: `p=2`, `r=2`, the split in all ten blocks, the structural shortcut on, the joint basis,
  dilations `{1,2,4}`, the post-fusion norm, 80 epochs at a constant 3e-5 after 2 warm-up epochs,
  and Euler with 50 steps.
- `scripts/param_audit.py` — accounts for the difference between the built model and the 62.9 M
  reported in the paper, and prices each remaining structural difference.
- `scripts/rho_s_null.py` — Monte-Carlo null distribution of the structural fraction `ρ_S`.
- `docs/paper_alignment.md` — item-by-item correspondence between the manuscript and the code.
- `tests/test_paper_alignment.py` — pins the published configurations to the reported settings and
  verifies that the new options leave the default model bit-identical.

### Changed

- `docs/main_structure_validation.json` was invalid JSON because of a trailing comma; fixed, with
  the limitations section restored and extended.
- `README.md` states the scope and the reproducibility status up front.
- `tests/test_documentation_assets.py` allows `docs/*.md`, since the alignment document is now part
  of the release.
- `tests/test_data_and_metrics.py` used `local_dilations` as its example of an unsupported config
  field; that is now a real option, so it uses a key that genuinely does not exist.

### Compatibility

All new options default to the previous behaviour. With default settings the state-dict keys, the
parameter count and the model output are bit-identical to v1.0.0, and `local_dilations: [1]`
reproduces the previous local branch exactly.

### Known gaps

- The parameter count does not match the paper: 55.840 M trainable, or 68.475 M including the
  non-trainable positional embedding, against 62.9 M reported. `scripts/param_audit.py` shows that
  no combination of the documented differences lands within 0.5 M.
- Because the basis is detached, the condition patch embedding of Eq. (3) receives no gradient and
  is frozen explicitly.
- The latent analysis of §3.1 (Table 1, Fig. 3), the tumour inpainting task (Tables 3–4), the
  external baselines, the routing-swap ablation and the MedicalNet LPIPS metric are not implemented.
- No run in this repository has reproduced a number printed in the manuscript.

## v1.0.0

Initial public release: latent translation pipeline for T1n→T1c and T2w→T2f on the frozen MAISI
codec, with P2/P4/P8 configurations and an optional structural shortcut.

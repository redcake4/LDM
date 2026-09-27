"""Structural-shortcut semantics, backwards compatibility, and persisted flags."""
from copy import deepcopy
import importlib
import json
import math
import sys
from types import MethodType

import pytest
import torch

from ldm.checkpoint import checked_payload
from ldm.config import DEFAULTS, load_config, model_contract, normalize_config, run_name
from ldm.layers import modulate
from ldm.mdsa import MDSA3D
from ldm.model import LDMBlock3D, LDMModel3D


def tiny_kwargs():
    return dict(patch_size=2, volume_size=(4, 8, 8), hidden_size=24, depth=3,
                global_heads=4, local_dim=6, bottleneck_dim=8, mdsa_blocks=(1,),
                mdsa_window=(2, 2, 2), mdsa_rank=2)


def make_block(**overrides):
    values = dict(hidden_size=12, global_heads=2, local_dim=6, inner_lr=1.0,
                  mlp_ratio=2.0, proj_drop=0.0, mdsa_mode="conditioned",
                  mdsa_window=(2, 2, 2), mdsa_rank=2, mdsa_gate_hidden=8)
    values.update(overrides)
    return LDMBlock3D(**values)


def reference_projection(features, reference, grid, window, rank):
    """Independent spatial loop oracle, including zero padding and cropping."""
    batch, _, channels = features.shape
    hgrid = features.reshape(batch, *grid, channels)
    rgrid = reference.reshape(batch, *grid, channels)
    output = torch.zeros_like(rgrid)
    for batch_index in range(batch):
        for d in range(0, grid[0], window[0]):
            for h in range(0, grid[1], window[1]):
                for w in range(0, grid[2], window[2]):
                    ends = tuple(min(start + size, extent)
                                 for start, size, extent in zip((d, h, w), window, grid))
                    lengths = tuple(end - start for end, start in zip(ends, (d, h, w)))
                    valid = tuple(slice(0, length) for length in lengths)
                    region = (batch_index, slice(d, ends[0]), slice(h, ends[1]), slice(w, ends[2]))
                    hw = torch.zeros(*window, channels, dtype=torch.float32)
                    rw = torch.zeros_like(hw)
                    hw[valid] = hgrid[region].detach().float()
                    rw[valid] = rgrid[region].float()
                    hw, rw = hw.reshape(-1, channels), rw.reshape(-1, channels)
                    _, vectors = torch.linalg.eigh(hw @ hw.T / channels)
                    basis = vectors[:, -rank:]
                    projected = (basis @ (basis.T @ rw)).reshape(*window, channels)
                    output[region] = projected[valid].to(reference.dtype)
    return output.reshape_as(reference)


@pytest.mark.parametrize("grid", [(2, 2, 2), (3, 3, 5)])
def test_reference_uses_feature_basis_once_and_preserves_branch_split(grid, monkeypatch):
    torch.manual_seed(81)
    features = torch.randn(2, math.prod(grid), 12)
    residual_stream = torch.randn_like(features) * 3 + 0.7
    mdsa = MDSA3D(12, window_size=(2, 2, 2), rank=2)
    expected_shortcut = reference_projection(features, residual_stream, grid, (2, 2, 2), 2)
    expected_structure = reference_projection(features, features, grid, (2, 2, 2), 2)
    wrong_basis = reference_projection(residual_stream, residual_stream, grid, (2, 2, 2), 2)
    old_structure, old_residual = mdsa.decompose(features, grid)
    original_eigh = torch.linalg.eigh
    calls = []

    def counted_eigh(*args, **kwargs):
        calls.append(args[0].shape)
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counted_eigh)
    structure, residual, shortcut = mdsa.decompose(features, grid, reference=residual_stream)
    assert len(calls) == 1, "The reference must reuse the branch projector"
    torch.testing.assert_close(structure, old_structure, atol=0, rtol=0)
    torch.testing.assert_close(residual, old_residual, atol=0, rtol=0)
    torch.testing.assert_close(structure, expected_structure, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(shortcut, expected_shortcut, atol=6e-6, rtol=3e-5)
    torch.testing.assert_close(structure + residual, features)
    assert not torch.allclose(shortcut, structure)
    assert not torch.allclose(shortcut, wrong_basis)


@pytest.mark.parametrize("mode", ["conditioned", "strict"])
def test_forward_reference_keeps_original_four_streams(mode):
    torch.manual_seed(15)
    mdsa = MDSA3D(12, mode=mode, window_size=(2, 2, 2), rank=2)
    features, reference = torch.randn(2, 8, 12), torch.randn(2, 8, 12)
    time = torch.randn(2, 12)
    old = mdsa(features, time, (2, 2, 2))
    new = mdsa(features, time, (2, 2, 2), reference=reference)
    assert len(old) == 4 and len(new) == 5
    for expected, actual in zip(old, new[:4]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(new[-1], reference_projection(features, reference, (2, 2, 2), (2, 2, 2), 2))


def test_detached_projector_preserves_reference_and_branch_gradients_with_padding():
    torch.manual_seed(71)
    grid = (3, 3, 5)
    features = torch.randn(1, math.prod(grid), 12, requires_grad=True)
    reference = torch.randn_like(features, requires_grad=True)
    mdsa = MDSA3D(12, window_size=(2, 2, 2), rank=2, detach_projector=True)
    structure, residual, shortcut = mdsa.decompose(features, grid, reference=reference)
    weight = torch.randn_like(shortcut)
    grad_h, grad_x = torch.autograd.grad((shortcut * weight).sum(), (features, reference),
                                       allow_unused=True, retain_graph=True)
    assert grad_h is None, "Detached projectors must not backpropagate through eigenvectors"
    oracle = reference_projection(features, reference, grid, (2, 2, 2), 2)
    expected_grad, = torch.autograd.grad((oracle * weight).sum(), reference)
    torch.testing.assert_close(grad_x, expected_grad, atol=4e-6, rtol=3e-5)
    assert grad_x.abs().sum() > 0 and torch.isfinite(grad_x).all()
    branch_grad, = torch.autograd.grad((structure + residual).square().sum(), features)
    torch.testing.assert_close(branch_grad, 2 * features)


def test_non_detached_reference_projector_remains_differentiable():
    torch.manual_seed(44)
    features = torch.randn(1, 4, 8, requires_grad=True)
    reference = torch.randn_like(features, requires_grad=True)
    mdsa = MDSA3D(8, window_size=(1, 2, 2), rank=2, detach_projector=False)
    shortcut = mdsa.decompose(features, (1, 2, 2), reference=reference)[2]
    gradients = torch.autograd.grad(shortcut.square().sum(), (features, reference))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in gradients)


@pytest.mark.parametrize("mode", ["conditioned", "strict"])
@pytest.mark.parametrize("main_structure", [False, True])
@pytest.mark.parametrize("mixer_gate,ffn_gate", [(0.0, 0.0), (0.37, -0.21)])
def test_block_uses_projected_raw_stream_then_unchanged_mixer_and_ffn(mode, main_structure, mixer_gate, ffn_gate):
    torch.manual_seed(200)
    block = make_block(mdsa_mode=mode, main_structure=main_structure)
    with torch.no_grad():
        block.adaLN_modulation[-1].weight.zero_()
        chunks = block.adaLN_modulation[-1].bias.chunk(6)
        chunks[0].copy_(torch.linspace(-0.5, 0.8, 12))
        chunks[1].copy_(torch.linspace(-0.2, 1.1, 12))
        chunks[2].fill_(mixer_gate)
        chunks[3].fill_(0.13)
        chunks[4].fill_(-0.17)
        chunks[5].fill_(ffn_gate)
    x = torch.randn(2, 8, 12, requires_grad=True)
    time = torch.randn(2, 12)
    shift, scale, alpha, shift_ffn, scale_ffn, beta = block.adaLN_modulation(time).chunk(6, -1)
    features = modulate(block.norm1(x), shift, scale)
    structure = reference_projection(features, x, (2, 2, 2), (2, 2, 2), 2)
    assert not torch.allclose(structure, reference_projection(features, features, (2, 2, 2), (2, 2, 2), 2))
    streams = block.mdsa(features, time, (2, 2, 2))
    mixed = block.mixer(features, (2, 2, 2), *streams)
    y = (structure if main_structure else x) + alpha[:, None] * mixed
    expected = y + beta[:, None] * block.mlp(modulate(block.norm2(y), shift_ffn, scale_ffn))
    actual = block(x, time, (2, 2, 2))
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=3e-5)
    if mixer_gate == ffn_gate == 0:
        torch.testing.assert_close(actual, structure if main_structure else x)
        assert torch.allclose(actual, x) is (not main_structure)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def legacy_block_forward(self, x, timestep_condition, grid_size):
    """Frozen pre-flag block equation, used with nonzero gates/output weights."""
    shift_mixer, scale_mixer, gate_mixer, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(timestep_condition).chunk(6, -1)
    features = modulate(self.norm1(x), shift_mixer, scale_mixer)
    streams = (None,) * 4 if self.mdsa is None else self.mdsa(features, timestep_condition, grid_size)
    mixed = self.mixer(features, grid_size, *streams)
    x = x + gate_mixer.unsqueeze(1) * mixed
    return x + gate_ffn.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_ffn, scale_ffn))


@pytest.mark.parametrize("mode", ["off", "gate_only", "conditioned", "strict"])
def test_default_and_false_preserve_seeded_state_rng_and_legacy_outputs(mode):
    values = tiny_kwargs() | {"mdsa_mode": mode}
    models = []
    states = []
    for options in ({}, {"main_structure": False}):
        torch.manual_seed(193)
        models.append(LDMModel3D(**values, **options))
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    for name, tensor in models[0].state_dict().items():
        torch.testing.assert_close(tensor, models[1].state_dict()[name], atol=0, rtol=0)
    # Initialization zeros the output head, so activate it before checking parity.
    torch.nn.init.normal_(models[0].final_layer.linear.weight, std=0.03)
    for block in models[0].blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.04)
    models[1].load_state_dict(models[0].state_dict())
    old = deepcopy(models[0])
    for block in old.blocks:
        block.forward = MethodType(legacy_block_forward, block)
    x, time = torch.randn(1, 8, 4, 8, 8), torch.tensor([0.42])
    expected = old(x, time)
    assert expected.abs().sum() > 0
    for model in models:
        torch.testing.assert_close(model(x, time), expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["conditioned", "strict"])
def test_structure_enables_only_configured_blocks_without_new_parameters(mode):
    torch.manual_seed(36)
    off = LDMModel3D(**(tiny_kwargs() | {"mdsa_mode": mode}))
    rng_off = torch.get_rng_state()
    torch.manual_seed(36)
    on = LDMModel3D(**(tiny_kwargs() | {"mdsa_mode": mode}), main_structure=True)
    assert torch.equal(rng_off, torch.get_rng_state())
    assert [block.main_structure for block in on.blocks] == [False, True, False]
    assert [block.mdsa is not None for block in on.blocks] == [False, True, False]
    assert off.state_dict().keys() == on.state_dict().keys()
    for name, tensor in off.state_dict().items():
        torch.testing.assert_close(tensor, on.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["off", "gate_only"])
def test_invalid_structure_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="conditioned or strict"):
        make_block(mdsa_mode=mode, main_structure=True)
    with pytest.raises(ValueError, match="conditioned or strict"):
        LDMModel3D(**(tiny_kwargs() | {"mdsa_mode": mode}), main_structure=True)
    if mode == "gate_only":
        module = MDSA3D(12, mode=mode)
        x = torch.randn(1, 8, 12)
        with pytest.raises(ValueError, match="decomposition"):
            module(x, None, (2, 2, 2), reference=x)


@pytest.mark.parametrize("flag", ["false", 0, 1, None])
def test_model_flag_requires_boolean(flag):
    with pytest.raises(ValueError, match="boolean"):
        LDMModel3D(**tiny_kwargs(), main_structure=flag)
    with pytest.raises(ValueError, match="boolean"):
        make_block(main_structure=flag)


def test_reference_shape_validation():
    module = MDSA3D(12)
    with pytest.raises(ValueError, match="shape and device"):
        module.decompose(torch.randn(1, 8, 12), (2, 2, 2), reference=torch.randn(1, 8, 11))


@pytest.mark.parametrize("yaml_flag", [False, True])
@pytest.mark.parametrize("cli_flag", [None, False, True])
def test_yaml_cli_precedence_contract_and_run_name(tmp_path, monkeypatch, capsys, yaml_flag, cli_flag):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  hidden_size: 24\n  depth: 2\n  global_heads: 4\n"
                    "  local_dim: 6\n  bottleneck_dim: 8\n  mdsa_blocks: [0, 1]\n"
                    f"  main_structure: {str(yaml_flag).lower()}\n", encoding="utf-8")
    args = ["train.py", "--config", str(path), "--dry-run", "--device", "cpu"]
    if cli_flag is not None:
        args.append("--main-structure" if cli_flag else "--no-main-structure")
    monkeypatch.setattr(sys, "argv", args)
    importlib.import_module("ldm.train").main()
    result = json.loads(capsys.readouterr().out)
    expected = yaml_flag if cli_flag is None else cli_flag
    assert result["config"]["model"]["main_structure"] is expected
    assert result["model_contract"]["model"]["main_structure"] is expected
    assert run_name(result["config"]).endswith("__main-structure") is expected


def test_missing_flag_normalizes_without_mutating_input_and_keeps_legacy_run_name():
    original = deepcopy(DEFAULTS)
    original["model"].pop("main_structure")
    old_name = "t1n_t1c__ldm__latent-p4__d1__mdsa-conditioned__seed0"
    assert normalize_config(original)["model"]["main_structure"] is False
    assert model_contract(original)["model"]["main_structure"] is False
    assert "main_structure" not in original["model"]
    assert run_name(original) == old_name
    assert load_config()["model"]["main_structure"] is False


@pytest.mark.parametrize("flag", ['"false"', "0", "1", "null"])
def test_yaml_rejects_non_boolean_flag(tmp_path, flag):
    path = tmp_path / "bad.yaml"
    path.write_text(f"model:\n  main_structure: {flag}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_config(path)


def checkpoint_payload(flag=False):
    config = deepcopy(DEFAULTS)
    config["model"]["main_structure"] = flag
    return {"format": "ldm_training_v1", "config": config, "model_contract": model_contract(config)}


@pytest.mark.parametrize("flag", [False, True])
def test_checkpoint_flag_round_trip_and_legacy_migration(tmp_path, flag):
    payload = checkpoint_payload(flag)
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    loaded = checked_payload(path)
    assert loaded["config"]["model"]["main_structure"] is flag
    assert loaded["model_contract"] == model_contract(loaded["config"])
    if not flag:
        payload["config"]["model"].pop("main_structure")
        payload["model_contract"]["model"].pop("main_structure")
        torch.save(payload, path)
        before = path.read_bytes()
        loaded = checked_payload(path)
        assert loaded["config"]["model"]["main_structure"] is False
        assert loaded["model_contract"]["model"]["main_structure"] is False
        assert path.read_bytes() == before, "Reading a legacy checkpoint must not rewrite it"


@pytest.mark.parametrize("case", ["config_only", "contract_only", "different_flags",
                                  "numeric_config", "numeric_contract", "legacy_extra_field"])
def test_checkpoint_rejects_mixed_schema_and_flag_tampering(tmp_path, case):
    payload = checkpoint_payload()
    if case == "config_only":
        payload["model_contract"]["model"].pop("main_structure")
    elif case == "contract_only":
        payload["config"]["model"].pop("main_structure")
    elif case == "different_flags":
        payload["model_contract"]["model"]["main_structure"] = True
    elif case == "numeric_config":
        payload["config"]["model"]["main_structure"] = 0
    elif case == "numeric_contract":
        payload["model_contract"]["model"]["main_structure"] = 0
    else:
        payload["config"]["model"].pop("main_structure")
        payload["model_contract"]["model"].pop("main_structure")
        payload["model_contract"]["unrecognized"] = True
    path = tmp_path / "tampered.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="boolean|inconsistent"):
        checked_payload(path)

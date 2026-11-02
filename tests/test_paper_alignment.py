"""The options added to match arXiv:2609.32794v1, and their backward compatibility."""
import pytest
import torch

from ldm.config import load_config, run_name, model_contract
from ldm.mdsa import MDSA3D
from ldm.mixer import LDMGlobalLocalMixer
from ldm.model import LDMModel3D

GRID = (2, 4, 4)
SMALL = dict(patch_size=8, hidden_size=64, depth=4, global_heads=4, local_dim=16,
             bottleneck_dim=16, mdsa_blocks=(1, 2))


def _model(**overrides):
    return LDMModel3D(**{**SMALL, **overrides})


def test_defaults_are_unchanged():
    """New options must not alter the released model's weights or outputs."""
    torch.manual_seed(0)
    reference = _model()
    torch.manual_seed(0)
    explicit = _model(mdsa_basis="state", local_dilations=(1,), post_fusion_norm=False)
    assert reference.state_dict().keys() == explicit.state_dict().keys()
    assert reference.c_embedder is None
    assert all(block.norm_fusion is None for block in reference.blocks)
    x, t = torch.randn(2, 8, 48, 64, 64), torch.rand(2)
    reference.eval(), explicit.eval()
    with torch.no_grad():
        assert torch.equal(reference(x, t), explicit(x, t))


def test_single_dilation_matches_the_previous_local_branch():
    torch.manual_seed(1)
    updated = LDMGlobalLocalMixer(64, 4, 16, 1.0, local_dilations=(1,))
    torch.manual_seed(1)
    original = LDMGlobalLocalMixer(64, 4, 16, 1.0)
    tokens = torch.randn(2, 32, 64)
    with torch.no_grad():
        assert torch.equal(updated(tokens, GRID), original(tokens, GRID))


@pytest.mark.parametrize("basis", ["condition", "joint"])
def test_condition_aware_basis_runs_and_is_frozen(basis):
    model = _model(mdsa_basis=basis, mdsa_rank=2, main_structure=True)
    assert model.c_embedder is not None
    assert not any(p.requires_grad for p in model.c_embedder.parameters()), (
        "the basis is detached, so the condition embedding can never receive gradient")
    out = model(torch.randn(2, 8, 48, 64, 64), torch.rand(2))
    assert out.shape == (2, 4, 48, 64, 64) and torch.isfinite(out).all()


def test_condition_changes_the_decomposition():
    mdsa = MDSA3D(32, rank=2, window_size=GRID, basis_source="joint")
    state = torch.randn(1, 32, 32)
    first, _ = mdsa.decompose(state, GRID, condition=torch.randn(1, 32, 32))
    second, _ = mdsa.decompose(state, GRID, condition=torch.randn(1, 32, 32))
    assert (first - second).abs().max() > 1e-6


def test_trace_normalized_gram_has_unit_trace():
    mdsa = MDSA3D(32, rank=2, window_size=GRID, basis_source="joint")
    gram = mdsa._normalized_gram(torch.randn(4, 32, 32))
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.ones_like(trace), atol=1e-5)


def test_condition_aware_basis_requires_a_condition():
    mdsa = MDSA3D(32, rank=2, window_size=GRID, basis_source="joint")
    with pytest.raises(ValueError, match="requires condition features"):
        mdsa.decompose(torch.randn(1, 32, 32), GRID)


def test_post_fusion_norm_is_identity_at_initialization():
    model = _model(post_fusion_norm=True).eval()
    assert all(block.norm_fusion is not None for block in model.blocks)
    with torch.no_grad():
        assert float(model(torch.randn(2, 8, 48, 64, 64), torch.rand(2)).abs().max()) == 0.0


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="mdsa_basis"):
        _model(mdsa_basis="bogus")
    with pytest.raises(ValueError, match="local_dilations"):
        LDMGlobalLocalMixer(64, 4, 16, 1.0, local_dilations=())
    with pytest.raises(ValueError, match="local_dilations"):
        LDMGlobalLocalMixer(64, 4, 16, 1.0, local_dilations=(0,))


@pytest.mark.parametrize("task", ["t1n_t1c", "t2w_t2f"])
def test_published_paper_configs_carry_the_reported_settings(task):
    """configs/paper_lsf_*.yaml must encode what the manuscript reports."""
    config = load_config(f"configs/paper_lsf_{task}.yaml")
    model, training, sampling = config["model"], config["training"], config["sampling"]
    assert config["task"] == task
    assert model["patch_size"] == 2                       # 24x32x32 token grid
    assert model["mdsa_rank"] == 2                        # Fig. 3
    assert list(model["mdsa_window"]) == [2, 4, 4]
    assert sorted(model["mdsa_blocks"]) == list(range(10))  # "every block"
    assert model["main_structure"] is True                 # Eq. (5)
    assert model["mdsa_basis"] == "joint"                  # Eq. (3)
    assert list(model["local_dilations"]) == [1, 2, 4]
    assert model["post_fusion_norm"] is True
    assert training["epochs"] == 80
    assert training["warmup_epochs"] == 2
    assert training["lr"] == training["min_lr"] == 3e-5    # constant after warm-up
    assert training["weight_decay"] == 0.0
    assert training["ema_decay"] == 0.9999
    assert training["batch_size"] == 1
    assert sampling["solver"] == "euler" and sampling["steps"] == 50
    assert "basis-joint" in run_name(config)
    assert model_contract(config)["model"]["mdsa_basis"] == "joint"


def test_paper_config_learning_rate_is_constant_after_warmup():
    from ldm.train import learning_rate
    config = load_config("configs/paper_lsf_t1n_t1c.yaml")
    after = {round(learning_rate(config, epoch), 12) for epoch in (2, 10, 40, 79)}
    assert after == {3e-5}
    assert learning_rate(config, 0) == 0.0

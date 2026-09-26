import pytest
import torch
from ldm.model import LDMModel3D
from ldm.mdsa import MDSA3D
from ldm.flow import LDMFlow, nfe
from ldm.config import load_config, ROOT


def tiny_model(patch=4):
    return LDMModel3D(patch_size=patch, volume_size=(8, 16, 16), hidden_size=24,
                     depth=2, global_heads=4, local_dim=6, bottleneck_dim=8,
                     mdsa_blocks=(0, 1), mdsa_rank=4)


@pytest.mark.parametrize("patch", [2, 4, 8])
def test_model_backward_and_single_scale(patch):
    model = tiny_model(patch)
    assert sum(block.mdsa is not None for block in model.blocks) == 2
    assert not any("extra" in name for name, _ in model.named_parameters())
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
    x = torch.randn(1, 8, 8, 16, 16, requires_grad=True)
    result = model(x, torch.tensor([0.5]))
    assert result.shape == (1, 4, 8, 16, 16)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for block in model.blocks:
        for weight in (block.mixer.w1, block.mixer.w2, block.mixer.w3):
            assert weight.grad is not None and torch.isfinite(weight.grad).all()
            assert weight.grad.abs().sum() > 0


@pytest.mark.parametrize("patch", [1, 16])
def test_patch_restriction(patch):
    with pytest.raises(ValueError, match="patch sizes"):
        tiny_model(patch)


def test_mdsa_decomposition_and_gate_initialization():
    module = MDSA3D(24, window_size=(2, 4, 4), rank=4)
    x = torch.randn(2, 32, 24, requires_grad=True)
    structure, residual, global_gate, local_gate = module(x, torch.randn(2, 24), (2, 4, 4))
    torch.testing.assert_close(structure+residual, x, atol=5e-7, rtol=1e-5)
    assert (structure * residual).sum().abs() < 5e-4
    torch.testing.assert_close(global_gate, torch.ones_like(global_gate))
    torch.testing.assert_close(local_gate, torch.ones_like(local_gate))
    (structure.square().mean() + residual.square().mean()).backward()
    assert torch.isfinite(x.grad).all()


def test_clean_prediction_loss_and_sampler():
    model = tiny_model()
    flow = LDMFlow(model)
    source = torch.randn(1, 4, 8, 16, 16)
    target = torch.randn_like(source)
    mask = torch.ones_like(source[:, :1])
    loss = flow(source, target, mask, t=torch.tensor([0.5]), noise=torch.zeros_like(source))
    torch.testing.assert_close(loss, 4 * target.square().mean())
    loss.backward()
    a = flow.generate(source, steps=3, generator=torch.Generator().manual_seed(2))
    b = flow.generate(source, steps=3, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(a, b)
    assert nfe(100, "heun") == 199
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("task", ["t1n_t1c", "t2w_t2f"])
@pytest.mark.parametrize("patch,tokens", [(2, 24576), (4, 3072), (8, 384)])
def test_release_configs(task, patch, tokens):
    cfg = load_config(ROOT / f"configs/{task}_p{patch}.yaml")
    model = LDMModel3D(**cfg["model"])
    assert model.x_embedder.num_patches == tokens
    assert model.blocks[4].mdsa is not None and model.blocks[5].mdsa is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_mixed_precision(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    model = tiny_model().cuda()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.01)
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.01)
    x = torch.randn(1, 8, 8, 16, 16, device="cuda")
    with torch.autocast("cuda", enabled=dtype != torch.float32, dtype=dtype):
        output = model(x, torch.tensor([0.5], device="cuda"))
    output.float().square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

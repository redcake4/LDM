"""Optional numerical lineage test; parent source is not a runtime dependency."""
import importlib
import os
from pathlib import Path
import pytest
import torch
from ldm.model import LDMModel3D
from ldm.flow import LDMFlow


@pytest.fixture
def parent(monkeypatch):
    root = os.environ.get("LDM_PARENT_ROOT")
    if not root:
        pytest.skip("Set LDM_PARENT_ROOT to the audited parent for optional parity tests")
    root = Path(root)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "frameworks/jit_ttt_3d"))
    return importlib.import_module("common.backbone")


@pytest.mark.parametrize("patch", [2, 4, 8])
@pytest.mark.parametrize("mdsa_mode", ["off", "conditioned"])
def test_parent_model_forward_and_gradients(parent, patch, mdsa_mode):
    kwargs = dict(volume_size=(8, 16, 16), patch_size=patch, hidden_size=24, depth=2,
                  bottleneck_dim=8, mdsa_mode=mdsa_mode, mdsa_blocks=(0, 1), mdsa_rank=4)
    reference = parent.TranslationJiT3D(**kwargs, in_channels=8, out_channels=4, num_heads=4,
        token_mixer="ttt", ttt_local_dilations=(1,), ttt_global_heads=4, ttt_local_dim=6,
        ttt_post_fusion="off", ttt_branch_gate="off")
    model = LDMModel3D(**kwargs, global_heads=4, local_dim=6)
    torch.manual_seed(4)
    with torch.no_grad():
        reference.final_layer.linear.weight.normal_(0, 0.02)
        for block in reference.blocks:
            block.adaLN_modulation[-1].weight.normal_(0, 0.02)
    mapping = {key.replace(".attn.", ".mixer."): value for key, value in reference.state_dict().items()}
    model.load_state_dict(mapping, strict=True)
    x1 = torch.randn(1, 8, 8, 16, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    time = torch.tensor([0.43])
    a, b = reference(x1, time), model(x2, time)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-7, rtol=1e-4)
    actual = dict(model.named_parameters())
    for key, param in reference.named_parameters():
        other = actual[key.replace(".attn.", ".mixer.")]
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("patch", [2, 4, 8])
def test_parent_flow_sampler(parent, patch):
    base = importlib.import_module("common.flow").FlowDenoiser

    class Reference(base):
        def __init__(self, net):
            torch.nn.Module.__init__(self)
            self.net, self.output_channels = net, 4
            self.noise_scale, self.t_eps, self.steps, self.method = 1.0, 0.05, 4, "heun"
            self.is_strict_self_inpaint = False
            self.patch_output_mode = "linear"

        def _net_predict(self, z, t, batch, return_pre_refiner=False):
            return self.net(torch.cat((z, batch["condition"]), 1), t.flatten())

    model = LDMModel3D(patch_size=patch, volume_size=(8, 16, 16), hidden_size=24, depth=2,
                      global_heads=4, local_dim=6, bottleneck_dim=8, mdsa_blocks=(0, 1), mdsa_rank=4)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    source = torch.randn(1, 4, 8, 16, 16)
    torch.manual_seed(44)
    expected = Reference(model).generate({"source": source, "condition": source})
    actual = LDMFlow(model).generate(source, steps=4, generator=torch.Generator().manual_seed(44))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

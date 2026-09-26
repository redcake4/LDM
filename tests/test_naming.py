import inspect
import torch
from ldm.layers import BottleneckPatchEmbed3D
from ldm.model import LDMBlock3D, LDMOutputHead, LDMModel3D
from scripts.audit_naming import audit_static


def test_runtime_naming_audit():
    assert audit_static()["issues"] == []


def test_projection_and_time_names():
    parameters = inspect.signature(BottleneckPatchEmbed3D).parameters
    assert "bottleneck_dim" in parameters and "pca_dim" not in parameters
    assert "timestep_condition" in inspect.signature(LDMBlock3D.forward).parameters
    assert "timestep_condition" in inspect.signature(LDMOutputHead.forward).parameters
    projection = BottleneckPatchEmbed3D((8, 8, 8), 4, 8, bottleneck_dim=6, embed_dim=24)
    assert set(projection.state_dict()) == {"proj1.weight", "proj2.weight", "proj2.bias"}


def test_existing_checkpoint_parameter_keys_retained():
    model = LDMModel3D(patch_size=4, volume_size=(8, 8, 8), hidden_size=24, depth=1,
        global_heads=4, local_dim=6, bottleneck_dim=8, mdsa_blocks=(0,), mdsa_rank=4)
    saved = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    assert "blocks.0.adaLN_modulation.1.weight" in saved
    assert "blocks.0.mixer.w3" in saved
    model.load_state_dict(saved, strict=True)
    with torch.no_grad():
        assert model(torch.zeros(1, 8, 8, 8, 8), torch.ones(1)).shape == (1, 4, 8, 8, 8)

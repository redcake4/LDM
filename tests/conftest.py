import h5py
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 2))
    yield
    torch.set_num_threads(old)


@pytest.fixture
def paired_h5(tmp_path):
    path = tmp_path / "paired.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update(task="t1n_t1c", source_modality="t1n", target_modality="t1c")
        for n, split in enumerate(("train", "val", "test")):
            group = f.create_group(split)
            source = np.zeros((1, 1, 8, 8, 8), np.float32)
            source[:, :, 1:7, 1:7, 1:7] = 0.2 + n * 0.05
            target = source.copy()
            target[:, :, 3:5, 3:5, 3:5] = 0.8
            mask = np.zeros((1, 3, 8, 8, 8), np.uint8)
            mask[:, 0, 3:5, 3:5, 3:5] = 1
            for key, value in (("source", source), ("target", target), ("mask", mask)):
                group.create_dataset(key, data=value)
            group.create_dataset("subject_id", data=[f"synthetic-{split}"], dtype=h5py.string_dtype())
    return path


class FakeAE(torch.nn.Module):
    """IO fixture only, never a replacement for the real MAISI in experiments."""
    def __init__(self, checkpoint=None, scale_factor=1.0):
        super().__init__()
        torch.randn(3)  # Model construction consumes RNG before pretrained load.
        self.scale_factor = scale_factor

    def set_scale_factor(self, scale):
        self.scale_factor = scale

    def encode_unscaled(self, x):
        v = torch.nn.functional.adaptive_avg_pool3d(x, (48, 64, 64))
        return torch.cat([v * (i + 1) for i in range(4)], 1)

    def encode(self, x):
        return self.encode_unscaled(x) * self.scale_factor

    def decode(self, x):
        return torch.nn.functional.interpolate((x[:, :1] / self.scale_factor).clamp(-1, 1), size=(160, 256, 256))

    @staticmethod
    def latent_valid_mask(valid):
        from ldm.autoencoder import FrozenMaisiAE
        return FrozenMaisiAE.latent_valid_mask(valid)

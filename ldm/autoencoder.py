"""Frozen NVIDIA MAISI-v1 adapter; no diffusion-model weights are used."""
from contextlib import nullcontext
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .runtime import sha256

AE_REPO = "nvidia/NV-Generate-CT"
AE_REVISION = "a481248bbd6462129efb2a3514e142bb95f87b25"
AE_FILENAME = "models/autoencoder_v1.pt"
AE_SHA256 = "1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3"
LATENT_SHAPE = (4, 48, 64, 64)
PREPROCESSING = "h5_0_1_endpad160_symmetric16_192_posterior_mean_v1"


def verify_ae(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAISI checkpoint missing: {path}. Run python scripts/download_ae.py")
    if sha256(path) != AE_SHA256:
        raise ValueError("MAISI SHA256 mismatch; only the pinned official checkpoint is supported")
    return path


class FrozenMaisiAE(nn.Module):
    def __init__(self, checkpoint, scale_factor=1.0):
        super().__init__()
        verify_ae(checkpoint)
        try:
            from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt (MONAI >= 1.5 is required)") from exc
        self.autoencoder = AutoencoderKlMaisi(spatial_dims=3, in_channels=1, out_channels=1,
            latent_channels=4, num_channels=(64, 128, 256), num_res_blocks=(2, 2, 2),
            norm_num_groups=32, norm_eps=1e-6, attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
            use_checkpointing=False, use_convtranspose=False, norm_float16=True,
            num_splits=4, dim_split=1)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.autoencoder.load_state_dict(payload.get("unet_state_dict", payload), strict=True)
        self.autoencoder.requires_grad_(False).eval()
        self.scale_factor = 1.0
        self.set_scale_factor(scale_factor)

    def train(self, mode=True):
        super().train(False)
        return self

    def set_scale_factor(self, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("AE scale must be finite and positive")
        self.scale_factor = float(scale)

    @staticmethod
    def pad(volume, value=-1.0):
        if tuple(volume.shape[1:]) != (1, 160, 256, 256):
            raise ValueError("Expected [B,1,160,256,256] after original-volume end padding")
        return F.pad(volume, (0, 0, 0, 0, 16, 16), value=value)

    def _context(self, tensor):
        # MAISI split normalization emits half activations on CUDA; convolutions
        # must stay within FP16 autocast even when the LDM model uses BF16/FP32.
        return torch.autocast("cuda", dtype=torch.float16) if tensor.is_cuda else nullcontext()

    @torch.no_grad()
    def encode_unscaled(self, volume):
        inputs = ((self.pad(volume) + 1) * 0.5).clamp(0, 1)
        with self._context(inputs):
            latent, _ = self.autoencoder.encode(inputs)
        if tuple(latent.shape[1:]) != LATENT_SHAPE:
            raise ValueError("MAISI latent shape mismatch")
        return latent.float()

    @torch.no_grad()
    def encode(self, volume):
        return self.encode_unscaled(volume) * self.scale_factor

    @torch.no_grad()
    def decode(self, latent):
        if tuple(latent.shape[1:]) != LATENT_SHAPE:
            raise ValueError("MAISI requires [B,4,48,64,64]")
        with self._context(latent):
            decoded = self.autoencoder.decode_stage_2_outputs(latent.float() / self.scale_factor)
        return (decoded.float().clamp(0, 1) * 2 - 1)[..., 16:176, :, :]

    @staticmethod
    def latent_valid_mask(valid):
        return F.avg_pool3d(FrozenMaisiAE.pad(valid, value=0.0), kernel_size=4, stride=4)

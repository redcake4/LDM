# Derived from the recorded parent implementation; see README.md#acknowledgements.
# References: SiT, Lightning-DiT; the timestep helper cites GLIDE.
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """RMS normalization over features with moments computed in FP32."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


def _to_3tuple(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"patch_size must have 3 values, got {value}.")
        return tuple(int(v) for v in value)
    return (int(value), int(value), int(value))


def _ceil_to_patch_multiple(volume_size, patch_size):
    return tuple(((int(v) + int(p) - 1) // int(p)) * int(p) for v, p in zip(volume_size, patch_size))


def _pad_3d_to_size(x, padded_size):
    d, h, w = (int(v) for v in x.shape[-3:])
    pd, ph, pw = (int(v) for v in padded_size)
    if pd < d or ph < h or pw < w:
        raise ValueError(f"padded_size {(pd, ph, pw)} must cover input size {(d, h, w)}.")
    pad_d, pad_h, pad_w = pd - d, ph - h, pw - w
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
    return x, (pad_d, pad_h, pad_w)


class BottleneckPatchEmbed3D(nn.Module):
    """Learned patch projection through a bottleneck, not fitted PCA."""

    def __init__(self, volume_size, patch_size, in_chans, bottleneck_dim, embed_dim, bias=True):
        super().__init__()
        self.original_volume_size = tuple(int(v) for v in volume_size)
        self.patch_size = _to_3tuple(patch_size)
        self.volume_size = _ceil_to_patch_multiple(self.original_volume_size, self.patch_size)
        self.pad_size = tuple(pv - ov for pv, ov in zip(self.volume_size, self.original_volume_size))
        self.grid_size = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.num_patches = int(np.prod(self.grid_size))
        self.last_grid_size = self.grid_size
        self.last_padded_size = self.volume_size
        self.last_pad_size = self.pad_size

        self.proj1 = nn.Conv3d(in_chans, bottleneck_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.proj2 = nn.Conv3d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        input_size = tuple(int(v) for v in x.shape[-3:])
        padded_size = _ceil_to_patch_multiple(input_size, self.patch_size)
        x, pad_size = _pad_3d_to_size(x, padded_size)
        x = self.proj2(self.proj1(x))
        self.last_grid_size = tuple(int(v) for v in x.shape[-3:])
        self.last_padded_size = padded_size
        self.last_pad_size = pad_size
        return x.flatten(2).transpose(1, 2)


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    d, h, w = [int(v) for v in grid_size]
    dim_d = (embed_dim // 3) // 2 * 2
    dim_h = (embed_dim // 3) // 2 * 2
    dim_w = embed_dim - dim_d - dim_h
    if dim_w % 2 != 0:
        dim_w -= 1
        dim_h += 1

    grid_d = np.arange(d, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    zz, yy, xx = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")
    emb_d = get_1d_sincos_pos_embed_from_grid(dim_d, zz)
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, yy)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, xx)
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)

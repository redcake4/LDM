"""Latent-only LDM with timestep conditioning and a single-scale local path."""
import torch
from torch import nn
from .layers import BottleneckPatchEmbed3D, RMSNorm, SwiGLUFFN, TimestepEmbedder, get_3d_sincos_pos_embed, modulate
from .mdsa import MDSA3D
from .mixer import LDMGlobalLocalMixer


class LDMBlock3D(nn.Module):
    def __init__(self, hidden_size, global_heads, local_dim, inner_lr, mlp_ratio,
                 proj_drop, mdsa_mode, mdsa_window, mdsa_rank, mdsa_gate_hidden):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.mixer = LDMGlobalLocalMixer(hidden_size, global_heads, local_dim, inner_lr)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.mdsa = None if mdsa_mode == "off" else MDSA3D(hidden_size, mode=mdsa_mode,
            window_size=mdsa_window, rank=mdsa_rank, gate_hidden=mdsa_gate_hidden, detach_projector=True)

    def forward(self, x, timestep_condition, grid_size):
        shift_mixer, scale_mixer, gate_mixer, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(timestep_condition).chunk(6, -1)
        features = modulate(self.norm1(x), shift_mixer, scale_mixer)
        streams = (None,) * 4 if self.mdsa is None else self.mdsa(features, timestep_condition, grid_size)
        mixed = self.mixer(features, grid_size, *streams)
        x = x + gate_mixer.unsqueeze(1) * mixed
        return x + gate_ffn.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_ffn, scale_ffn))


class LDMOutputHead(nn.Module):
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size ** 3 * 4)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, timestep_condition):
        shift, scale = self.adaLN_modulation(timestep_condition).chunk(2, 1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class LDMModel3D(nn.Module):
    def __init__(self, patch_size=4, volume_size=(48, 64, 64), hidden_size=512, depth=10,
                 global_heads=8, local_dim=64, bottleneck_dim=96, mlp_ratio=4.0,
                 inner_lr=1.0, proj_drop=0.0, mdsa_mode="conditioned", mdsa_window=(2, 4, 4),
                 mdsa_rank=8, mdsa_blocks=(4, 5), mdsa_gate_hidden=32):
        super().__init__()
        if patch_size not in (2, 4, 8):
            raise ValueError("LDM supports latent patch sizes 2, 4 and 8 only")
        if hidden_size < 6 or hidden_size % 2 or depth <= 0:
            raise ValueError("LDM requires an even hidden size >= 6 and positive depth")
        if any(v <= 0 or v % patch_size for v in volume_size):
            raise ValueError("Latent dimensions must be positive patch multiples")
        if mdsa_mode not in {"off", "conditioned", "strict", "gate_only"}:
            raise ValueError("Unsupported MDSA mode")
        if len(set(mdsa_blocks)) != len(mdsa_blocks) or any(i < 0 or i >= depth for i in mdsa_blocks):
            raise ValueError("MDSA blocks must be unique zero-based indices within depth")
        if mdsa_mode != "off" and not mdsa_blocks:
            raise ValueError("Enabled MDSA requires at least one active block")
        self.volume_size, self.patch_size = tuple(volume_size), patch_size
        self.hidden_size = hidden_size
        self.x_embedder = BottleneckPatchEmbed3D(volume_size, patch_size, 8, bottleneck_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([LDMBlock3D(hidden_size, global_heads, local_dim, inner_lr,
            mlp_ratio, proj_drop if depth // 4 <= i < depth // 4 * 3 else 0.0,
            mdsa_mode if i in mdsa_blocks else "off", mdsa_window, mdsa_rank, mdsa_gate_hidden) for i in range(depth)])
        self.final_layer = LDMOutputHead(hidden_size, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        position = get_3d_sincos_pos_embed(self.hidden_size, self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(position).float().unsqueeze(0))
        for index in (0, 2):
            nn.init.normal_(self.t_embedder.mlp[index].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            if block.mdsa is not None:
                block.mdsa.reset_gates()
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x, t):
        if x.ndim != 5 or x.shape[1] != 8 or tuple(x.shape[-3:]) != self.volume_size:
            raise ValueError(f"Expected concatenated state/source [B,8,{self.volume_size}], got {tuple(x.shape)}")
        timestep_condition = self.t_embedder(t.flatten())
        x = self.x_embedder(x)
        x = x + self.pos_embed.to(x.dtype)
        grid = self.x_embedder.grid_size
        for block in self.blocks:
            x = block(x, timestep_condition, grid)
        x = self.final_layer(x, timestep_condition)
        d, h, w = grid
        p = self.patch_size
        x = x.reshape(x.shape[0], d, h, w, p, p, p, 4)
        return torch.einsum("ndhwpqrc->ncdphqwr", x).reshape(x.shape[0], 4, *self.volume_size)

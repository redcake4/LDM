"""Single-scale adaptive global/local mixer. See README.md#acknowledgements for lineage."""
import torch
from torch import nn
from torch.nn import functional as F


class LDMGlobalLocalMixer(nn.Module):
    def __init__(self, dim, num_heads=8, local_dim=64, inner_lr=1.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads or local_dim <= 0 or inner_lr < 0:
            raise ValueError("Invalid global heads, local width or inner learning rate")
        self.dim, self.num_heads, self.local_dim = dim, num_heads, local_dim
        self.head_dim, self.inner_lr = dim // num_heads, float(inner_lr)
        self.scale = 27 ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim + 3 * local_dim)
        self.w1 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, self.head_dim))
        self.w2 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, self.head_dim))
        self.w3 = nn.Parameter(torch.empty(local_dim, 1, 3, 3, 3))
        for weight in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(weight, std=0.02)
        self.proj = nn.Linear(dim + local_dim, dim)

    def global_update(self, k, v):
        z1, z2 = k @ self.w1, k @ self.w2
        sigmoid = torch.sigmoid(z2)
        error = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (error * z2 * sigmoid)
        g2 = k.transpose(-2, -1) @ (error * z1 * (sigmoid * (1 + z2 * (1 - sigmoid))))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1)
        return self.w1 - self.inner_lr * g1, self.w2 - self.inner_lr * g2

    def local_update(self, k, v):
        batch, channels, depth, height, width = k.shape
        error = -v / float(depth * height * width) * self.scale
        padded = F.pad(k, (1, 1, 1, 1, 1, 1))
        # Analytic inner gradient over the fixed 3x3x3 support, still differentiable.
        terms = []
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    terms.append((padded[:, :, z:z+depth, y:y+height, x:x+width] * error).sum((-3, -2, -1)))
        grad = torch.stack(terms, -1).reshape(batch * channels, 1, 3, 3, 3)
        grad = grad / (torch.linalg.vector_norm(grad, dim=(-3, -2, -1), keepdim=True) + 1)
        return self.w3.repeat(batch, 1, 1, 1, 1) - self.inner_lr * grad

    def forward(self, x, grid_size, global_x=None, local_x=None, global_gate=None, local_gate=None):
        batch, tokens, dim = x.shape
        depth, height, width = grid_size
        if tokens != depth * height * width or dim != self.dim:
            raise ValueError("Token shape does not match the LDM grid")
        if global_x is None and local_x is None:
            projected = self.qkv(x)
        else:
            global_x = x if global_x is None else global_x
            local_x = x if local_x is None else local_x
            if global_x.shape != x.shape or local_x.shape != x.shape:
                raise ValueError("MDSA streams must match the input shape")
            split = 3 * dim
            projected = torch.cat((F.linear(global_x, self.qkv.weight[:split], self.qkv.bias[:split]),
                                   F.linear(local_x, self.qkv.weight[split:], self.qkv.bias[split:])), -1)
        q1, k1, v1, q2, k2, v2 = torch.split(projected, [dim] * 3 + [self.local_dim] * 3, -1)
        q1, k1, v1 = [v.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
                      for v in (q1, k1, v1)]
        q2, k2, v2 = [v.reshape(batch, depth, height, width, self.local_dim).permute(0, 4, 1, 2, 3)
                      for v in (q2, k2, v2)]
        w1, w2 = self.global_update(k1, v1)
        global_out = ((q1 @ w1) * F.silu(q1 @ w2)).transpose(1, 2).reshape(batch, tokens, dim)
        kernel = self.local_update(k2, v2)
        local_out = F.conv3d(q2.reshape(1, batch * self.local_dim, depth, height, width),
                             kernel, padding=1, groups=batch * self.local_dim)
        local_out = local_out.reshape(batch, self.local_dim, tokens).transpose(1, 2)
        if global_gate is not None:
            global_out = global_out * global_gate.to(global_out.dtype)
        if local_gate is not None:
            local_out = local_out * local_gate.to(local_out.dtype)
        return self.proj(torch.cat((global_out, local_out), -1))

# Derived from the recorded parent implementation; see README.md#acknowledgements.
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3tuple(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"Expected three spatial values, got {value}.")
        return tuple(int(v) for v in value)
    return (int(value), int(value), int(value))


class MDSA3D(nn.Module):
    """Windowed low-rank structure/residual decomposition for 3D token grids."""

    MODES = {"gate_only", "strict", "conditioned"}

    def __init__(
        self,
        dim,
        mode="conditioned",
        window_size=(2, 4, 4),
        rank=8,
        gate_hidden=32,
        detach_projector=True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(f"MDSA mode must be one of {sorted(self.MODES)}, got {mode!r}.")
        self.window_size = _to_3tuple(window_size)
        if any(value <= 0 for value in self.window_size):
            raise ValueError("MDSA window dimensions must be positive.")
        self.window_tokens = int(self.window_size[0] * self.window_size[1] * self.window_size[2])
        self.rank = int(rank)
        if self.rank <= 0 or self.rank > self.window_tokens:
            raise ValueError(
                f"MDSA rank must be in [1, {self.window_tokens}], got {self.rank}."
            )
        self.detach_projector = bool(detach_projector)
        gate_hidden = int(gate_hidden)
        if gate_hidden <= 0:
            raise ValueError("MDSA gate_hidden must be positive.")

        if self.mode == "strict":
            self.context_proj = None
            self.time_proj = None
            self.energy_proj = None
            self.gate_out = None
        else:
            self.context_proj = nn.Linear(self.dim, gate_hidden, bias=False)
            self.time_proj = nn.Linear(self.dim, gate_hidden, bias=False)
            self.energy_proj = nn.Linear(2, gate_hidden, bias=True)
            self.gate_out = nn.Linear(gate_hidden, 2, bias=True)
            self.reset_gates()

    def reset_gates(self):
        if self.gate_out is not None:
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)

    def _partition(self, x, grid_size):
        batch, tokens, channels = x.shape
        depth, height, width = _to_3tuple(grid_size)
        if tokens != depth * height * width:
            raise ValueError(
                f"Token count {tokens} does not match MDSA grid size {(depth, height, width)}."
            )
        wd, wh, ww = self.window_size
        pad_d = (-depth) % wd
        pad_h = (-height) % wh
        pad_w = (-width) % ww
        grid = x.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3)
        if pad_d or pad_h or pad_w:
            grid = F.pad(grid, (0, pad_w, 0, pad_h, 0, pad_d))
        padded = (depth + pad_d, height + pad_h, width + pad_w)
        pd, ph, pw = padded
        windows = (
            grid.permute(0, 2, 3, 4, 1)
            .reshape(batch, pd // wd, wd, ph // wh, wh, pw // ww, ww, channels)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(-1, self.window_tokens, channels)
        )
        return windows, padded

    def _reverse(self, windows, batch, grid_size, padded_size):
        depth, height, width = _to_3tuple(grid_size)
        pd, ph, pw = _to_3tuple(padded_size)
        wd, wh, ww = self.window_size
        channels = windows.shape[-1]
        grid = (
            windows.reshape(batch, pd // wd, ph // wh, pw // ww, wd, wh, ww, channels)
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .reshape(batch, pd, ph, pw, channels)
        )
        return grid[:, :depth, :height, :width].reshape(batch, depth * height * width, channels)

    def decompose(self, x, grid_size, reference=None):
        """Split x; optionally project a reference using exactly the same basis.

        The optional third result is P_x @ reference, not P_reference @ reference.
        This keeps the structural shortcut in the residual stream's coordinates
        while the mixer branches use normalized/modulated features.
        """
        if reference is not None and (reference.shape != x.shape or reference.device != x.device):
            raise ValueError("MDSA reference must match the feature shape and device")
        batch = x.shape[0]
        windows, padded_size = self._partition(x, grid_size)
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            work = windows.float()
            gram_input = work.detach() if self.detach_projector else work
            gram = torch.matmul(gram_input, gram_input.transpose(-2, -1)) / float(self.dim)
            if self.detach_projector:
                with torch.no_grad():
                    _, eigenvectors = torch.linalg.eigh(gram)
                    basis = eigenvectors[..., -self.rank:].detach()
            else:
                _, eigenvectors = torch.linalg.eigh(gram)
                basis = eigenvectors[..., -self.rank:]
            coefficients = torch.matmul(basis.transpose(-2, -1), work)
            structure_windows = torch.matmul(basis, coefficients)
            residual_windows = work - structure_windows
            if reference is not None:
                reference_windows, _ = self._partition(reference, grid_size)
                reference_structure = torch.matmul(
                    basis, torch.matmul(basis.transpose(-2, -1), reference_windows.float())
                )

        structure = self._reverse(
            structure_windows.to(dtype=x.dtype), batch, grid_size, padded_size
        )
        residual = self._reverse(
            residual_windows.to(dtype=x.dtype), batch, grid_size, padded_size
        )
        if reference is None:
            return structure, residual
        shortcut = self._reverse(
            reference_structure.to(dtype=reference.dtype), batch, grid_size, padded_size
        )
        return structure, residual, shortcut

    def _gates(self, context, structure, residual, timestep_condition):
        structure_energy = structure.float().square().mean(dim=-1, keepdim=True)
        residual_energy = residual.float().square().mean(dim=-1, keepdim=True)
        energy = torch.log1p(torch.cat([structure_energy, residual_energy], dim=-1))
        energy = energy.to(dtype=context.dtype)
        if timestep_condition is None:
            time_feature = torch.zeros_like(self.context_proj(context[:, :1]))
        else:
            time_feature = self.time_proj(timestep_condition).unsqueeze(1)
        hidden = self.context_proj(context) + time_feature + self.energy_proj(energy)
        gates = 2.0 * torch.sigmoid(self.gate_out(F.silu(hidden)))
        return gates[..., :1], gates[..., 1:]

    def forward(self, x, timestep_condition, grid_size, reference=None):
        """Return branch inputs/gates and, if requested, a structural shortcut."""
        if self.mode == "gate_only":
            if reference is not None:
                raise ValueError("A structural shortcut requires MDSA decomposition")
            structure = x
            residual = x
        else:
            decomposition = self.decompose(x, grid_size, reference=reference)
            structure, residual = decomposition[:2]

        if self.mode == "strict":
            global_gate = None
            local_gate = None
        else:
            global_gate, local_gate = self._gates(
                structure, structure, residual, timestep_condition
            )
        streams = (structure, residual, global_gate, local_gate)
        return streams if reference is None else (*streams, decomposition[2])

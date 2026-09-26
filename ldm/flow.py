"""Clean-latent prediction, velocity matching and the inherited Heun sampler."""
import torch
from torch import nn


class LDMFlow(nn.Module):
    def __init__(self, net, time_mean=-0.8, time_std=0.8, time_epsilon=0.05, noise_scale=1.0):
        super().__init__()
        self.net = net
        if time_std < 0 or not 0 < time_epsilon <= 1 or noise_scale <= 0:
            raise ValueError("Invalid flow parameters")
        self.time_mean, self.time_std = time_mean, time_std
        self.time_epsilon, self.noise_scale = time_epsilon, noise_scale

    def forward(self, source, target, valid_mask, t=None, noise=None):
        if source.shape != target.shape:
            raise ValueError("Source and target latent shapes must match")
        if t is None:
            t = torch.sigmoid(torch.randn(target.shape[0], device=target.device) * self.time_std + self.time_mean)
        t = t.reshape(-1, 1, 1, 1, 1)
        noise = torch.randn_like(target) * self.noise_scale if noise is None else noise
        state = t * target + (1 - t) * noise
        denominator = (1 - t).clamp_min(self.time_epsilon)
        desired = (target - state) / denominator
        clean = self.net(torch.cat((state, source), 1), t.flatten())
        predicted = (clean.float() - state) / denominator
        error = (desired - predicted).square()
        mask = valid_mask.to(error).expand_as(error)
        return ((error * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()

    @torch.no_grad()
    def generate(self, source, steps=100, solver="heun", generator=None):
        if steps <= 0 or solver not in {"heun", "euler"}:
            raise ValueError("Positive steps and heun/euler solver required")
        state = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator) * self.noise_scale
        times = torch.linspace(0, 1, steps + 1, device=source.device)

        def velocity(value, time):
            t = time.expand(source.shape[0])
            clean = self.net(torch.cat((value, source), 1), t).float()
            return (clean - value) / (1 - time).clamp_min(self.time_epsilon)

        for i in range(steps):
            dt = times[i + 1] - times[i]
            first = velocity(state, times[i])
            candidate = state + dt * first
            # Last interval uses Euler, exactly as in the parent sampler.
            state = state + dt * 0.5 * (first + velocity(candidate, times[i + 1])) if solver == "heun" and i < steps - 1 else candidate
        return state


def nfe(steps, solver):
    return 2 * steps - 1 if solver == "heun" else steps

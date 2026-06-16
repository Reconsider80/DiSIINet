"""DDIM noise schedule and sampling utilities."""

import math
from typing import Optional

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    """Cosine / linear noise schedule for DDIM diffusion."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps

        if schedule == "cosine":
            alphas_cumprod = self._cosine_schedule(num_train_timesteps)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    @staticmethod
    def _cosine_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from Nichol & Dhariwal (2021), Eq. 22 in paper."""
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        f_t = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        return alphas_cumprod[1:]

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def get_inference_timesteps(
        self, num_inference_steps: int, device: torch.device
    ) -> torch.Tensor:
        """Select descending timestep subsequence for DDIM accelerated sampling."""
        return torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        ).long()

    @torch.no_grad()
    def ddim_sample_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single DDIM reverse step (Eq. 5 and 11 in paper)."""
        alpha_t = self.alphas_cumprod[t]
        if t_prev >= 0:
            alpha_prev = self.alphas_cumprod[t_prev]
        else:
            alpha_prev = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        pred_x0 = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

        sigma_t = (
            eta
            * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
            * torch.sqrt(1 - alpha_t / alpha_prev)
        )

        dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * noise_pred
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        if eta > 0 and noise is not None:
            x_prev = x_prev + sigma_t * noise

        return x_prev

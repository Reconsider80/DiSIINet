"""DiSIINet: Dual-branch DDIM with Symbiotic Information Interaction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.schedule import DiffusionSchedule
from .sii import SymbioticInformationInteraction
from .unet import ConditionalUNet
from .vae import (
    SegDecoder,
    SegEncoder,
    SegImageEncoder,
    VAEDecoder,
    VAEEncoder,
)


class DiSIINet(nn.Module):
    """
    Diffusion-based Symbiotic Information Interaction Network.

    Two branches:
      - DiEnh: image enhancement via DDIM in latent space
      - DiSeg: segmentation via DDIM in latent space
    Linked by SII module for bidirectional feature exchange.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        num_heads: int = 8,
        dropout: float = 0.0,
        num_train_timesteps: int = 1000,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_channels = latent_channels

        # DiEnh branch encoders/decoders
        self.image_encoder = VAEEncoder(
            in_channels, latent_channels, base_channels, channel_mult
        )
        self.image_decoder = VAEDecoder(
            in_channels, latent_channels, base_channels, channel_mult
        )

        # DiSeg branch encoders/decoders
        self.seg_encoder = SegEncoder(
            num_classes, latent_channels, base_channels // 2, channel_mult
        )
        self.seg_decoder = SegDecoder(
            num_classes, latent_channels, base_channels // 2, channel_mult
        )
        self.seg_image_encoder = SegImageEncoder(in_channels, base_channels)

        # SII module
        self.sii = SymbioticInformationInteraction(
            latent_channels, num_heads, dropout
        )

        # Denoising UNets
        self.unet_enh = ConditionalUNet(
            in_channels=latent_channels,
            cond_channels=latent_channels,
            sii_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.unet_seg = ConditionalUNet(
            in_channels=latent_channels,
            cond_channels=base_channels,
            sii_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.schedule = DiffusionSchedule(
            num_train_timesteps=num_train_timesteps,
            schedule=schedule,
        )

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)

    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return self.seg_encoder(mask)

    def forward(
        self,
        x_lq: torch.Tensor,
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with symbiotic interaction.

        Args:
            x_lq: low-quality input image (B, 1, H, W)
            x_gt: ground-truth high-quality image (B, 1, H, W)
            mask_gt: one-hot segmentation mask (B, C, H, W)
        """
        batch_size = x_lq.shape[0]
        device = x_lq.device
        t = torch.randint(
            0,
            self.schedule.num_train_timesteps,
            (batch_size,),
            device=device,
        )

        z_lq = self.encode_image(x_lq)
        z_gt_enh = self.encode_image(x_gt)
        z_gt_seg = self.encode_mask(mask_gt)
        f_lq = self.seg_image_encoder(x_lq)

        noise_enh = torch.randn_like(z_gt_enh)
        noise_seg = torch.randn_like(z_gt_seg)

        z_enh_t = self.schedule.add_noise(z_gt_enh, noise_enh, t)
        z_seg_t = self.schedule.add_noise(z_gt_seg, noise_seg, t)

        f_oi, f_os = self.sii(z_enh_t, z_seg_t)

        eps_enh_pred = self.unet_enh(z_enh_t, t, z_lq, f_oi)
        eps_seg_pred = self.unet_seg(z_seg_t, t, f_lq, f_os)

        z_enh_0 = self._predict_x0(z_enh_t, eps_enh_pred, t)
        z_seg_0 = self._predict_x0(z_seg_t, eps_seg_pred, t)

        ip = self.image_decoder(z_enh_0)
        sp_logits = self.seg_decoder(z_seg_0)
        sp = torch.sigmoid(sp_logits)

        return {
            "eps_enh_pred": eps_enh_pred,
            "eps_enh_target": noise_enh,
            "eps_seg_pred": eps_seg_pred,
            "eps_seg_target": noise_seg,
            "ip": ip,
            "sp": sp,
            "sp_logits": sp_logits,
        }

    def _predict_x0(
        self, x_t: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )
        return (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha

    @torch.no_grad()
    def sample(
        self,
        x_lq: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """
        DDIM inference with SII at every step (Sec. 3.1, 4.3).
        """
        device = x_lq.device
        z_lq = self.encode_image(x_lq)
        f_lq = self.seg_image_encoder(x_lq)

        b, c, h, w = z_lq.shape
        z_enh = torch.randn(b, c, h, w, device=device)
        z_seg = torch.randn(b, c, h, w, device=device)

        timesteps = self.schedule.get_inference_timesteps(
            num_inference_steps, device
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1

            f_oi, f_os = self.sii(z_enh, z_seg)

            eps_enh = self.unet_enh(z_enh, t_batch, z_lq, f_oi)
            eps_seg = self.unet_seg(z_seg, t_batch, f_lq, f_os)

            noise_enh = torch.randn_like(z_enh) if eta > 0 else None
            noise_seg = torch.randn_like(z_seg) if eta > 0 else None

            z_enh = self.schedule.ddim_sample_step(
                z_enh, eps_enh, t.item(), t_prev, eta, noise_enh
            )
            z_seg = self.schedule.ddim_sample_step(
                z_seg, eps_seg, t.item(), t_prev, eta, noise_seg
            )

        ip = self.image_decoder(z_enh)
        sp_logits = self.seg_decoder(z_seg)
        sp = torch.sigmoid(sp_logits)

        return {"ip": ip, "sp": sp, "sp_logits": sp_logits}

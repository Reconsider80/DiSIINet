"""Symbiotic Information Interaction (SII) module."""

import torch
import torch.nn as nn


class SymbioticInformationInteraction(nn.Module):
    """
    SII module with Enh-Controller and Seg-Controller (Sec. 3.2).

    Enh-Controller: Q=Z_seg, K=V=Z_enh -> F_oi for DiEnh
    Seg-Controller: Q=Z_enh, K=V=Z_seg -> F_os for DiSeg
    """

    def __init__(
        self,
        latent_channels: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_heads = num_heads
        head_dim = latent_channels // num_heads
        assert latent_channels % num_heads == 0

        self.enh_controller = nn.MultiheadAttention(
            embed_dim=latent_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.seg_controller = nn.MultiheadAttention(
            embed_dim=latent_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _flatten_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)"""
        b, c, h, w = x.shape
        return x.view(b, c, h * w).permute(0, 2, 1)

    def _unflatten_spatial(
        self, x: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """(B, H*W, C) -> (B, C, H, W)"""
        b, _, c = x.shape
        return x.permute(0, 2, 1).view(b, c, h, w)

    def forward(
        self,
        z_enh: torch.Tensor,
        z_seg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_enh: enhancement latent at timestep t, (B, C, H, W)
            z_seg: segmentation latent at timestep t, (B, C, H, W)

        Returns:
            f_oi: guided feature for DiEnh branch (Eq. 14)
            f_os: guided feature for DiSeg branch (Eq. 16)
        """
        _, _, h, w = z_enh.shape

        z_enh_flat = self._flatten_spatial(z_enh)
        z_seg_flat = self._flatten_spatial(z_seg)

        # Enh-Controller: Q=Z_seg, K=V=Z_enh (Eq. 13)
        f_oic, _ = self.enh_controller(z_seg_flat, z_enh_flat, z_enh_flat)
        f_oic = self._unflatten_spatial(f_oic, h, w)
        f_oi = torch.relu(f_oic + z_enh)

        # Seg-Controller: Q=Z_enh, K=V=Z_seg (Eq. 15)
        f_osc, _ = self.seg_controller(z_enh_flat, z_seg_flat, z_seg_flat)
        f_osc = self._unflatten_spatial(f_osc, h, w)
        f_os = torch.relu(f_osc + z_seg)

        return f_oi, f_os

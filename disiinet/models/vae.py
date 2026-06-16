"""VAE encoder/decoder for latent-space diffusion."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class VAEEncoder(nn.Module):
    """Image encoder mapping RGB/grayscale images to latent space."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        blocks = []
        ch = base_channels
        for mult in channel_mult:
            out_ch = base_channels * mult
            blocks.extend([ResBlock(ch), ResBlock(ch)])
            blocks.append(Downsample(ch))
            if ch != out_ch:
                blocks.append(nn.Conv2d(ch, out_ch, 1))
            ch = out_ch

        self.down = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, latent_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.down(h)
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        moments = self.conv_out(h)
        mean, logvar = moments.chunk(2, dim=1)
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)


class VAEDecoder(nn.Module):
    """Decoder mapping latent representations back to image space."""

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        ch = base_channels * channel_mult[-1]
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)

        blocks = []
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            blocks.extend([ResBlock(ch), ResBlock(ch)])
            blocks.append(Upsample(ch))
            if ch != out_ch:
                blocks.append(nn.Conv2d(ch, out_ch, 1))
            ch = out_ch

        self.up = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.up(h)
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        return self.conv_out(h)


class SegEncoder(nn.Module):
    """Encoder for segmentation masks (one-hot or multi-class)."""

    def __init__(
        self,
        num_classes: int = 4,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_mult: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(num_classes, base_channels, 3, padding=1)

        blocks = []
        ch = base_channels
        for mult in channel_mult:
            out_ch = base_channels * mult
            blocks.extend([ResBlock(ch), ResBlock(ch)])
            blocks.append(Downsample(ch))
            if ch != out_ch:
                blocks.append(nn.Conv2d(ch, out_ch, 1))
            ch = out_ch

        self.down = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, latent_channels, 3, padding=1)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(mask)
        h = self.down(h)
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        return self.conv_out(h)


class SegDecoder(nn.Module):
    """Decoder for segmentation latent to mask logits."""

    def __init__(
        self,
        num_classes: int = 4,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_mult: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        ch = base_channels * channel_mult[-1]
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)

        blocks = []
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            blocks.extend([ResBlock(ch), ResBlock(ch)])
            blocks.append(Upsample(ch))
            if ch != out_ch:
                blocks.append(nn.Conv2d(ch, out_ch, 1))
            ch = out_ch

        self.up = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, num_classes, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.up(h)
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        return self.conv_out(h)


class SegImageEncoder(nn.Module):
    """Feature encoder for LQ images used as conditioning in DiSeg branch."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        base_channels: int = 32,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            ResBlock(base_channels),
            Downsample(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 1),
            ResBlock(base_channels * 2),
            nn.GroupNorm(min(8, base_channels * 2), base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

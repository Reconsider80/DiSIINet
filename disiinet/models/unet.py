"""Conditional UNet for DDIM noise prediction."""

import math

import torch
import torch.nn as nn


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlockCond(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = nn.functional.silu(h)
        h = h + self.emb_proj(nn.functional.silu(emb))[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = nn.functional.silu(h)
        h = self.dropout(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x).view(b, c, h * w).permute(0, 2, 1)
        out, _ = self.attn(flat, flat, flat)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        num_res: int,
        use_attn: bool,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_res):
            layers.append(ResBlockCond(ch, out_ch, emb_dim, dropout))
            ch = out_ch
            if use_attn:
                layers.append(AttentionBlock(out_ch, num_heads))
        self.layers = nn.ModuleList(layers)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, list]:
        skips = []
        for layer in self.layers:
            if isinstance(layer, ResBlockCond):
                x = layer(x, emb)
            else:
                x = layer(x)
            skips.append(x)
        return self.down(x), skips


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        num_res: int,
        use_attn: bool,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        layers = []
        ch = out_ch * 2
        for _ in range(num_res):
            layers.append(ResBlockCond(ch, out_ch, emb_dim, dropout))
            ch = out_ch
            if use_attn:
                layers.append(AttentionBlock(out_ch, num_heads))
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, skips: list, emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        for layer in self.layers:
            if isinstance(layer, ResBlockCond):
                skip = skips.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(
                        x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                    )
                x = layer(torch.cat([x, skip], dim=1), emb)
            else:
                x = layer(x)
        return x


class ConditionalUNet(nn.Module):
    """
    UNet for noise prediction in DDIM framework.
    Conditions on LQ features and SII guidance (Eq. 4, 10).
    """

    def __init__(
        self,
        in_channels: int = 4,
        cond_channels: int = 4,
        sii_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        total_in = in_channels + cond_channels + sii_channels
        self.conv_in = nn.Conv2d(total_in, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = base_channels
        resolution = 128
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            use_attn = resolution in attention_resolutions
            self.down_blocks.append(
                DownBlock(
                    ch, out_ch, time_dim, num_res_blocks,
                    use_attn, num_heads, dropout,
                )
            )
            ch = out_ch
            resolution //= 2

        self.mid1 = ResBlockCond(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid2 = ResBlockCond(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        resolution = 128 // (2 ** len(channel_mult))
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            use_attn = resolution in attention_resolutions
            self.up_blocks.append(
                UpBlock(
                    ch, out_ch, time_dim, num_res_blocks,
                    use_attn, num_heads, dropout,
                )
            )
            ch = out_ch
            resolution *= 2

        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        sii_feat: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(get_timestep_embedding(t, self.time_mlp[0].in_features))

        if cond.shape[-2:] != x.shape[-2:]:
            cond = nn.functional.interpolate(
                cond, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if sii_feat.shape[-2:] != x.shape[-2:]:
            sii_feat = nn.functional.interpolate(
                sii_feat, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        h = self.conv_in(torch.cat([x, cond, sii_feat], dim=1))
        all_skips = [h]

        for block in self.down_blocks:
            h, skips = block(h, t_emb)
            all_skips.extend(skips)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for block in self.up_blocks:
            h = block(h, all_skips, t_emb)

        h = self.norm_out(h)
        h = nn.functional.silu(h)
        return self.conv_out(h)

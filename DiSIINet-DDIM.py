import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Union
from einops import rearrange, repeat

# ==================== DDIM专用组件 ====================
class DDIMNoiseSchedule(nn.Module):
    """DDIM噪声调度器，支持确定性和随机性采样"""
    def __init__(self, num_timesteps=1000, schedule='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if schedule == 'linear':
            # DDIM通常使用线性或cosine调度
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule == 'cosine':
            # Cosine调度（改进版本）
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.register_buffer('betas', betas)
        
        # 预计算所有参数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 扩散过程参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # DDIM反转和采样参数
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # 后验方差参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # DDIM的sigma参数
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

class DDIMScheduler:
    """DDIM调度器，管理采样和反转过程"""
    def __init__(self, num_timesteps=1000, schedule='linear', ddim_discretize='uniform', ddim_eta=0.0):
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.ddim_eta = ddim_eta
        self.noise_schedule = DDIMNoiseSchedule(num_timesteps, schedule)
        
        # DDIM子序列
        if ddim_discretize == 'uniform':
            self.ddim_timesteps = torch.arange(0, num_timesteps, num_timesteps // 50)
        else:
            raise ValueError(f"Unknown discretize method: {ddim_discretize}")
        
        self.ddim_timesteps_prev = torch.cat([torch.tensor([-1]), self.ddim_timesteps[:-1]])
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始图像"""
        sqrt_recip_alphas_cumprod_t = self.noise_schedule.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.noise_schedule.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def ddim_step(self, x_t, t, t_prev, predicted_noise, eta=0.0):
        """DDIM单步采样"""
        # 获取参数
        alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t] ** 2
        alphas_cumprod_t_prev = self.noise_schedule.sqrt_alphas_cumprod[t_prev] ** 2 if t_prev >= 0 else 1.0
        
        sqrt_alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # 预测x_0
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        # 计算方向
        sigma_t = eta * torch.sqrt((1 - alphas_cumprod_t_prev) / (1 - alphas_cumprod_t)) * \
                  torch.sqrt(1 - alphas_cumprod_t / alphas_cumprod_t_prev)
        
        dir_xt = torch.sqrt(1 - alphas_cumprod_t_prev - sigma_t ** 2) * predicted_noise
        
        x_prev = torch.sqrt(alphas_cumprod_t_prev) * pred_x0 + dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev, pred_x0
    
    def ddim_reverse_step(self, x_t, t, t_next, predicted_noise, eta=0.0):
        """DDIM反转单步（用于编辑和插值）"""
        # 获取参数
        alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t] ** 2
        alphas_cumprod_t_next = self.noise_schedule.sqrt_alphas_cumprod[t_next] ** 2
        
        sqrt_alphas_cumprod_t = self.noise_schedule.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # 预测x_0
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        # 计算方向（反转）
        sigma_t_next = eta * torch.sqrt((1 - alphas_cumprod_t_next) / (1 - alphas_cumprod_t)) * \
                      torch.sqrt(1 - alphas_cumprod_t / alphas_cumprod_t_next)
        
        dir_xt_next = torch.sqrt(1 - alphas_cumprod_t_next - sigma_t_next ** 2) * predicted_noise
        
        x_next = torch.sqrt(alphas_cumprod_t_next) * pred_x0 + dir_xt_next
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_next = x_next + sigma_t_next * noise
        
        return x_next

# ==================== 基础网络组件（保持与DDPM相同） ====================
class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('b h d i, b h d j -> b h i j', q * scale, k)
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)

class Downsample(nn.Module):
    """下采样"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """上采样"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DiffusionUNet(nn.Module):
    """基础扩散UNet（与DDPM相同）"""
    def __init__(self, in_channels, out_channels, base_channels=64, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径
        self.down1 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels, time_emb_dim),
            AttentionBlock(base_channels),
            ResidualBlock(base_channels, base_channels, time_emb_dim)
        ])
        self.downsample1 = Downsample(base_channels)
        
        self.down2 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim),
            AttentionBlock(base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        ])
        self.downsample2 = Downsample(base_channels * 2)
        
        self.down3 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        ])
        self.downsample3 = Downsample(base_channels * 4)
        
        # 瓶颈层
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 8, time_emb_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        ])
        
        # 上采样路径
        self.upsample3 = Upsample(base_channels * 4)
        self.up3 = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim),
            AttentionBlock(base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        ])
        
        self.upsample2 = Upsample(base_channels * 2)
        self.up2 = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim),
            AttentionBlock(base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        ])
        
        self.upsample1 = Upsample(base_channels)
        self.up1 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim),
            AttentionBlock(base_channels),
            ResidualBlock(base_channels, base_channels, time_emb_dim)
        ])
        
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        t = self.time_mlp(timesteps)
        
        x = self.init_conv(x)
        
        # 下采样
        skip1 = []
        for layer in self.down1:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            skip1.append(x)
        x = self.downsample1(x)
        
        skip2 = []
        for layer in self.down2:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            skip2.append(x)
        x = self.downsample2(x)
        
        skip3 = []
        for layer in self.down3:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            skip3.append(x)
        x = self.downsample3(x)
        
        # 瓶颈层
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        # 上采样
        x = self.upsample3(x)
        for i, layer in enumerate(self.up3):
            if i == 0:
                x = torch.cat([x, skip3[-1-i]], dim=1)
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        x = self.upsample2(x)
        for i, layer in enumerate(self.up2):
            if i == 0:
                x = torch.cat([x, skip2[-1-i]], dim=1)
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        x = self.upsample1(x)
        for i, layer in enumerate(self.up1):
            if i == 0:
                x = torch.cat([x, skip1[-1-i]], dim=1)
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        return self.out(x)

# ==================== 共生信息交互模块 (SII) ====================
class CrossAttentionFusion(nn.Module):
    """跨注意力融合模块"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj = nn.Conv2d(channels, channels, 1)
        self.value_proj = nn.Conv2d(channels, channels, 1)
        
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        
        feat1_norm = self.norm1(feat1)
        feat2_norm = self.norm2(feat2)
        
        Q = self.query_proj(feat1_norm)
        K = self.key_proj(feat2_norm)
        V = self.value_proj(feat2_norm)
        
        Q = rearrange(Q, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        K = rearrange(K, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        V = rearrange(V, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        
        scale = self.head_dim ** -0.5
        attention = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, V)
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=H, ww=W)
        
        out = self.out_proj(out)
        out = feat1 + out
        
        return out

class SymbioticInformationInteraction(nn.Module):
    """共生信息交互模块 (SII)"""
    def __init__(self, enh_channels, seg_channels, fusion_channels=256):
        super().__init__()
        
        self.enh_conv = nn.Conv2d(enh_channels, fusion_channels, 1)
        self.seg_conv = nn.Conv2d(seg_channels, fusion_channels, 1)
        
        self.enh_to_seg_attention = CrossAttentionFusion(fusion_channels)
        self.seg_to_enh_attention = CrossAttentionFusion(fusion_channels)
        
        self.enh_enhance = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 3, padding=1),
            nn.GroupNorm(8, fusion_channels),
            nn.SiLU(),
            nn.Conv2d(fusion_channels, enh_channels, 1)
        )
        
        self.seg_enhance = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 3, padding=1),
            nn.GroupNorm(8, fusion_channels),
            nn.SiLU(),
            nn.Conv2d(fusion_channels, seg_channels, 1)
        )
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1),
            nn.SiLU(),
            nn.Conv2d(fusion_channels, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, enh_feat, seg_feat):
        enh_adj = self.enh_conv(enh_feat)
        seg_adj = self.seg_conv(seg_feat)
        
        seg_influenced = self.enh_to_seg_attention(seg_adj, enh_adj)
        enh_influenced = self.seg_to_enh_attention(enh_adj, seg_adj)
        
        combined = torch.cat([enh_influenced, seg_influenced], dim=1)
        gates = self.gate_net(combined)
        gate_enh, gate_seg = gates.chunk(2, dim=1)
        
        fused_enh = enh_influenced * gate_enh + seg_influenced * (1 - gate_enh)
        fused_seg = seg_influenced * gate_seg + enh_influenced * (1 - gate_seg)
        
        enh_out = self.enh_enhance(torch.cat([fused_enh, enh_adj], dim=1)) + enh_feat
        seg_out = self.seg_enhance(torch.cat([fused_seg, seg_adj], dim=1)) + seg_feat
        
        return enh_out, seg_out

# ==================== 基于DDIM的双分支网络 ====================
class DiEnhBranch(nn.Module):
    """基于DDIM的图像增强分支 (DiEnh)"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.diffusion_unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim
        )
        
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(16, base_channels * 2),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.GroupNorm(32, base_channels * 4),
                nn.SiLU()
            )
        ])
        
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, timesteps, return_features=True):
        enhanced = self.diffusion_unet(x, timesteps)
        
        if return_features:
            B, C, H, W = x.shape
            features = []
            
            for i, extractor in enumerate(self.feature_extractors):
                if i == 0:
                    feat = F.interpolate(x, scale_factor=1/(2**i), mode='bilinear')
                else:
                    feat = F.avg_pool2d(features[-1], 2)
                feat = extractor(feat)
                features.append(feat)
            
            quality_score = self.quality_head(features[-1])
            
            return enhanced, features, quality_score
        else:
            return enhanced

class DiSegBranch(nn.Module):
    """基于DDIM的图像分割分支 (DiSeg)"""
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.n_classes = n_classes
        
        self.diffusion_unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=n_classes,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim
        )
        
        self.boundary_aware = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(16, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, n_classes, 1)
        )
        
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(16, base_channels * 2),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.GroupNorm(32, base_channels * 4),
                nn.SiLU()
            )
        ])
    
    def forward(self, x, timesteps, return_features=True):
        segmentation = self.diffusion_unet(x, timesteps)
        
        if return_features:
            B, C, H, W = x.shape
            features = []
            
            for i, extractor in enumerate(self.feature_extractors):
                if i == 0:
                    feat = F.interpolate(x, scale_factor=1/(2**i), mode='bilinear')
                else:
                    feat = F.avg_pool2d(features[-1], 2)
                feat = extractor(feat)
                features.append(feat)
            
            boundary = self.boundary_aware(features[-1])
            boundary = F.interpolate(boundary, size=(H, W), mode='bilinear')
            
            return segmentation, boundary, features
        else:
            return segmentation

# ==================== 完整的DDIM-DiSIINet ====================
class DDIMDiSIINet(nn.Module):
    """基于DDIM的共生信息交互网络 (DiSIINet)"""
    def __init__(self, 
                 in_channels=3, 
                 n_classes=1, 
                 base_channels=64, 
                 time_emb_dim=256, 
                 num_timesteps=1000,
                 schedule='linear',
                 ddim_discretize='uniform',
                 ddim_eta=0.0):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.ddim_eta = ddim_eta
        
        # 初始化DDIM调度器
        self.scheduler = DDIMScheduler(
            num_timesteps=num_timesteps,
            schedule=schedule,
            ddim_discretize=ddim_discretize,
            ddim_eta=ddim_eta
        )
        
        # 双扩散分支
        self.enh_branch = DiEnhBranch(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim
        )
        
        self.seg_branch = DiSegBranch(
            in_channels=in_channels,
            n_classes=n_classes,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim
        )
        
        # 共生信息交互模块
        self.sii_modules = nn.ModuleList([
            SymbioticInformationInteraction(
                enh_channels=base_channels * (2**i),
                seg_channels=base_channels * (2**i),
                fusion_channels=base_channels * (2**i)
            )
            for i in range(3)
        ])
        
        # 时间步编码器
        self.timestep_encoder = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 联合优化头
        self.joint_optimizer = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.GroupNorm(32, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(16, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, 1, 1)
        )
        
        # DDIM特定组件
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, noisy_image, timesteps, return_all=False):
        B = noisy_image.shape[0]
        
        # 编码时间步
        t_emb = self.timestep_encoder(timesteps)
        
        # 双分支前向传播
        enhanced_init, enh_features_init, quality_score = self.enh_branch(noisy_image, timesteps)
        segmented_init, boundary_init, seg_features_init = self.seg_branch(noisy_image, timesteps)
        
        # 多层级共生信息交互
        enh_features = []
        seg_features = []
        
        for i, sii_module in enumerate(self.sii_modules):
            enh_feat = enh_features_init[i] if i < len(enh_features_init) else enh_features[-1]
            seg_feat = seg_features_init[i] if i < len(seg_features_init) else seg_features[-1]
            
            enh_feat_interacted, seg_feat_interacted = sii_module(enh_feat, seg_feat)
            
            enh_features.append(enh_feat_interacted)
            seg_features.append(seg_feat_interacted)
        
        # 使用交互后的特征进行最终预测
        combined_feat = torch.cat([enh_features[-1], seg_features[-1]], dim=1)
        joint_output = self.joint_optimizer(combined_feat)
        
        # 最终输出
        enhanced_final = enhanced_init + joint_output
        segmented_final = segmented_init + F.interpolate(joint_output, size=segmented_init.shape[2:])
        
        if return_all:
            return {
                'enhanced': enhanced_final,
                'segmented': segmented_final,
                'boundary': boundary_init,
                'quality_score': quality_score,
                'enhanced_init': enhanced_init,
                'segmented_init': segmented_init,
                'enh_features': enh_features,
                'seg_features': seg_features
            }
        else:
            return enhanced_final, segmented_final
    
    def predict_noise(self, x_t, t):
        """预测噪声（DDIM的核心）"""
        # 双分支预测
        enhanced, segmented = self(x_t, t, return_all=False)
        
        # 使用增强分支的输出作为噪声预测
        # 在DDIM中，网络预测的是噪声ε
        return enhanced
    
    def ddim_sample_step(self, x_t, t, t_prev, eta=None):
        """DDIM单步采样"""
        if eta is None:
            eta = self.ddim_eta
        
        # 预测噪声
        predicted_noise = self.predict_noise(x_t, t)
        
        # DDIM采样步骤
        x_prev, pred_x0 = self.scheduler.ddim_step(
            x_t, t, t_prev, predicted_noise, eta=eta
        )
        
        return x_prev, pred_x0
    
    @torch.no_grad()
    def ddim_sample(self, 
                   shape, 
                   num_inference_steps=50,
                   eta=0.0,
                   guidance_scale=3.0,
                   return_intermediates=False):
        """DDIM采样生成图像"""
        device = next(self.parameters()).device
        b, c, h, w = shape
        
        # DDIM子序列
        timesteps = self.scheduler.ddim_timesteps[:num_inference_steps]
        timesteps_prev = self.scheduler.ddim_timesteps_prev[:num_inference_steps]
        
        # 初始化噪声
        x_t = torch.randn(shape, device=device)
        
        intermediates = []
        
        for i, (t, t_prev) in enumerate(zip(reversed(timesteps), reversed(timesteps_prev))):
            # 创建时间步张量
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # DDIM采样步骤
            x_t, pred_x0 = self.ddim_sample_step(x_t, t_batch, t_prev, eta=eta)
            
            if return_intermediates and i % 5 == 0:
                intermediates.append(pred_x0)
        
        # 最终去噪
        enhanced, segmented = self(x_t, torch.zeros_like(t_batch))
        
        if return_intermediates:
            return enhanced, segmented, intermediates
        return enhanced, segmented
    
    @torch.no_grad()
    def ddim_inversion(self, x_0, num_inference_steps=50):
        """DDIM反转（图像到噪声）"""
        device = x_0.device
        b = x_0.shape[0]
        
        timesteps = self.scheduler.ddim_timesteps[:num_inference_steps]
        timesteps_next = self.scheduler.ddim_timesteps[1:num_inference_steps+1]
        
        x_t = x_0
        trajectory = [x_0]
        
        for t, t_next in zip(timesteps, timesteps_next):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.predict_noise(x_t, t_batch)
            
            # DDIM反转步骤
            x_next = self.scheduler.ddim_reverse_step(
                x_t, t_batch, t_next, predicted_noise, eta=self.ddim_eta
            )
            
            x_t = x_next
            trajectory.append(x_t)
        
        return x_t, trajectory
    
    def p_losses(self, x_start, seg_gt, t, noise=None):
        """计算DDIM损失（与DDPM相同）"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 添加噪声
        x_noisy = self.scheduler.q_sample(x_start, t, noise)
        
        # 网络预测
        enhanced, segmented = self(x_noisy, t)
        
        # 重建损失
        recon_loss = F.mse_loss(enhanced, noise)  # DDIM预测噪声
        
        # 分割损失
        if seg_gt is not None:
            seg_loss = F.binary_cross_entropy_with_logits(segmented, seg_gt)
        else:
            seg_loss = torch.tensor(0.0, device=x_start.device)
        
        # 总损失
        total_loss = recon_loss + seg_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'seg_loss': seg_loss,
            'x_noisy': x_noisy,
            'enhanced': enhanced,
            'segmented': segmented
        }
    
    def sample_timesteps(self, batch_size, device):
        """采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

# ==================== 训练和损失函数 ====================
class DDIMDiSIINetLoss(nn.Module):
    """DDIM-DiSIINet的损失函数"""
    def __init__(self, 
                 lambda_recon=1.0, 
                 lambda_seg=1.0, 
                 lambda_boundary=0.5, 
                 lambda_quality=0.1,
                 lambda_consistency=0.2):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_seg = lambda_seg
        self.lambda_boundary = lambda_boundary
        self.lambda_quality = lambda_quality
        self.lambda_consistency = lambda_consistency
    
    def forward(self, predictions, targets):
        clean_image = targets['clean_image']
        seg_gt = targets['segmentation_gt']
        
        # 重建损失（DDIM预测噪声）
        recon_loss = F.mse_loss(predictions['enhanced'], targets.get('noise', torch.randn_like(predictions['enhanced'])))
        
        # 分割损失
        seg_loss = F.binary_cross_entropy_with_logits(predictions['segmented'], seg_gt)
        
        # 边界损失
        if 'boundary' in predictions:
            seg_gt_boundary = self._compute_boundary(seg_gt)
            boundary_loss = F.mse_loss(predictions['boundary'], seg_gt_boundary)
        else:
            boundary_loss = torch.tensor(0.0, device=clean_image.device)
        
        # 质量一致性损失
        if 'quality_score' in predictions:
            seg_accuracy = self._compute_seg_accuracy(predictions['segmented'], seg_gt)
            quality_loss = F.mse_loss(predictions['quality_score'], seg_accuracy.unsqueeze(1))
        else:
            quality_loss = torch.tensor(0.0, device=clean_image.device)
        
        # DDIM一致性损失
        consistency_loss = self._compute_ddim_consistency_loss(predictions, targets)
        
        # 总损失
        total_loss = (self.lambda_recon * recon_loss +
                     self.lambda_seg * seg_loss +
                     self.lambda_boundary * boundary_loss +
                     self.lambda_quality * quality_loss +
                     self.lambda_consistency * consistency_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'seg_loss': seg_loss,
            'boundary_loss': boundary_loss,
            'quality_loss': quality_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_boundary(self, seg_map):
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                             dtype=torch.float32, device=seg_map.device).view(1, 1, 3, 3)
        boundary = F.conv2d(seg_map, kernel, padding=1)
        boundary = torch.abs(boundary)
        boundary = torch.clamp(boundary, 0, 1)
        return boundary
    
    def _compute_seg_accuracy(self, pred, gt):
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        correct = (pred_binary == gt).float()
        accuracy = correct.mean(dim=(1, 2, 3))
        return accuracy
    
    def _compute_ddim_consistency_loss(self, predictions, targets):
        """DDIM一致性损失"""
        # 确保DDIM采样的确定性
        loss = 0.0
        
        if 'enhanced_init' in predictions and 'enhanced' in predictions:
            # 初始预测和最终预测应保持一致
            enhanced_consistency = F.mse_loss(predictions['enhanced'], predictions['enhanced_init'])
            loss += enhanced_consistency
        
        if 'segmented_init' in predictions and 'segmented' in predictions:
            segmented_consistency = F.mse_loss(predictions['segmented'], predictions['segmented_init'])
            loss += segmented_consistency
        
        return loss

# ==================== 训练循环示例 ====================
class DDIMTrainer:
    """DDIM训练器"""
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(device)
        
    def train_step(self, batch):
        """单训练步"""
        self.model.train()
        
        # 获取数据
        clean_images = batch['clean_image'].to(self.device)
        seg_gts = batch['segmentation_gt'].to(self.device)
        
        # 采样时间步
        batch_size = clean_images.shape[0]
        timesteps = self.model.sample_timesteps(batch_size, self.device)
        
        # 采样噪声
        noise = torch.randn_like(clean_images)
        
        # 前向传播
        self.optimizer.zero_grad()
        
        # DDIM训练损失
        loss_dict = self.model.p_losses(clean_images, seg_gts, timesteps, noise)
        
        # 计算总损失
        predictions = {
            'enhanced': loss_dict['enhanced'],
            'segmented': loss_dict['segmented'],
            'noise': noise
        }
        
        targets = {
            'clean_image': clean_images,
            'segmentation_gt': seg_gts,
            'noise': noise
        }
        
        losses = self.loss_fn(predictions, targets)
        
        # 反向传播
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            clean_images = batch['clean_image'].to(self.device)
            seg_gts = batch['segmentation_gt'].to(self.device)
            
            batch_size = clean_images.shape[0]
            timesteps = self.model.sample_timesteps(batch_size, self.device)
            noise = torch.randn_like(clean_images)
            
            # 前向传播
            loss_dict = self.model.p_losses(clean_images, seg_gts, timesteps, noise)
            
            predictions = {
                'enhanced': loss_dict['enhanced'],
                'segmented': loss_dict['segmented']
            }
            
            targets = {
                'clean_image': clean_images,
                'segmentation_gt': seg_gts
            }
            
            losses = self.loss_fn(predictions, targets)
            total_loss += losses['total_loss'].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def generate_samples(self, num_samples=4, image_size=128):
        """生成样本"""
        self.model.eval()
        
        shape = (num_samples, 3, image_size, image_size)
        
        # DDIM采样
        enhanced, segmented = self.model.ddim_sample(
            shape, 
            num_inference_steps=50,
            eta=self.model.ddim_eta
        )
        
        return enhanced, segmented

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 DDIM-DiSIINet 网络")
    print("=" * 60)
    
    # 创建DDIM模型
    model = DDIMDiSIINet(
        in_channels=3,
        n_classes=1,
        base_channels=32,
        time_emb_dim=128,
        num_timesteps=100,
        schedule='linear',
        ddim_discretize='uniform',
        ddim_eta=0.0
    )
    
    # 测试输入
    batch_size = 2
    image = torch.randn(batch_size, 3, 128, 128)
    seg_gt = torch.randn(batch_size, 1, 128, 128)
    
    print(f"输入图像形状: {image.shape}")
    print(f"分割GT形状: {seg_gt.shape}")
    
    # 采样时间步
    timesteps = model.sample_timesteps(batch_size, image.device)
    print(f"采样时间步: {timesteps}")
    
    # 前向传播
    enhanced, segmented = model(image, timesteps)
    print(f"增强图像形状: {enhanced.shape}")
    print(f"分割结果形状: {segmented.shape}")
    
    # 测试DDIM采样
    print("\n测试DDIM采样...")
    enhanced_sample, segmented_sample = model.ddim_sample(
        shape=(batch_size, 3, 128, 128),
        num_inference_steps=20
    )
    print(f"DDIM采样增强图像形状: {enhanced_sample.shape}")
    print(f"DDIM采样分割结果形状: {segmented_sample.shape}")
    
    # 测试DDIM反转
    print("\n测试DDIM反转...")
    x_t, trajectory = model.ddim_inversion(image, num_inference_steps=20)
    print(f"反转后噪声形状: {x_t.shape}")
    print(f"反转轨迹长度: {len(trajectory)}")
    
    # 计算损失
    targets = {
        'clean_image': torch.randn_like(image),
        'segmentation_gt': seg_gt
    }
    
    predictions = {
        'enhanced': enhanced,
        'segmented': segmented,
        'enhanced_init': enhanced,
        'segmented_init': segmented
    }
    
    loss_fn = DDIMDiSIINetLoss()
    losses = loss_fn(predictions, targets)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024**2:.2f} MB")
    
    # 测试训练步骤
    print("\n测试训练步骤...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DDIMTrainer(model, loss_fn, optimizer, device='cpu')
    
    batch = {
        'clean_image': torch.randn(batch_size, 3, 128, 128),
        'segmentation_gt': torch.randn(batch_size, 1, 128, 128)
    }
    
    train_losses = trainer.train_step(batch)
    print("训练损失:", train_losses['total_loss'])
    
    print("\nDDIM-DiSIINet测试完成!")

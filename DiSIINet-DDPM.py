import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from einops import rearrange, repeat

# ==================== 基础扩散组件 ====================
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
        
        # 时间嵌入投影
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        
        # 添加时间嵌入
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
        
        # 生成QKV
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # 注意力计算
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('b h d i, b h d j -> b h i j', q * scale, k)
        attn = attn.softmax(dim=-1)
        
        # 聚合
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
    """基础扩散UNet"""
    def __init__(self, in_channels, out_channels, base_channels=64, time_emb_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
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
        # 时间嵌入
        t = self.time_mlp(timesteps)
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 下采样
        # Level 1
        skip1 = []
        for layer in self.down1:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            skip1.append(x)
        x = self.downsample1(x)
        
        # Level 2
        skip2 = []
        for layer in self.down2:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            skip2.append(x)
        x = self.downsample2(x)
        
        # Level 3
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
        # Level 3
        x = self.upsample3(x)
        for i, layer in enumerate(self.up3):
            if i == 0:
                x = torch.cat([x, skip3[-1-i]], dim=1)
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        # Level 2
        x = self.upsample2(x)
        for i, layer in enumerate(self.up2):
            if i == 0:
                x = torch.cat([x, skip2[-1-i]], dim=1)
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        # Level 1
        x = self.upsample1(x)
        for i, layer in enumerate(self.up1):
            if i == 0:
                x = torch.cat([x, skip1[-1-i]], dim=1)
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        # 输出
        return self.out(x)

# ==================== 共生信息交互模块 (SII) ====================
class CrossAttentionFusion(nn.Module):
    """跨注意力融合模块"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        # Query, Key, Value投影
        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj = nn.Conv2d(channels, channels, 1)
        self.value_proj = nn.Conv2d(channels, channels, 1)
        
        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # 层归一化
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, feat1, feat2):
        """
        feat1: 来自分支1的特征 [B, C, H, W]
        feat2: 来自分支2的特征 [B, C, H, W]
        """
        B, C, H, W = feat1.shape
        
        # 归一化
        feat1_norm = self.norm1(feat1)
        feat2_norm = self.norm2(feat2)
        
        # 生成Q, K, V
        Q = self.query_proj(feat1_norm)  # Query来自分支1
        K = self.key_proj(feat2_norm)    # Key来自分支2
        V = self.value_proj(feat2_norm)  # Value来自分支2
        
        # 重塑为多头注意力格式
        Q = rearrange(Q, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        K = rearrange(K, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        V = rearrange(V, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        
        # 注意力计算
        scale = self.head_dim ** -0.5
        attention = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attention = F.softmax(attention, dim=-1)
        
        # 加权聚合
        out = torch.matmul(attention, V)
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=H, ww=W)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 残差连接
        out = feat1 + out
        
        return out

class SymbioticInformationInteraction(nn.Module):
    """共生信息交互模块 (SII)"""
    def __init__(self, enh_channels, seg_channels, fusion_channels=256):
        super().__init__()
        
        # 通道调整
        self.enh_conv = nn.Conv2d(enh_channels, fusion_channels, 1)
        self.seg_conv = nn.Conv2d(seg_channels, fusion_channels, 1)
        
        # 双向跨注意力融合
        self.enh_to_seg_attention = CrossAttentionFusion(fusion_channels)
        self.seg_to_enh_attention = CrossAttentionFusion(fusion_channels)
        
        # 特征增强
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
        
        # 门控机制
        self.gate_net = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1),
            nn.SiLU(),
            nn.Conv2d(fusion_channels, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, enh_feat, seg_feat):
        """
        enh_feat: 增强分支特征 [B, C1, H, W]
        seg_feat: 分割分支特征 [B, C2, H, W]
        """
        # 通道调整
        enh_adj = self.enh_conv(enh_feat)
        seg_adj = self.seg_conv(seg_feat)
        
        # 双向信息交互
        # 增强→分割
        seg_influenced = self.enh_to_seg_attention(seg_adj, enh_adj)
        
        # 分割→增强
        enh_influenced = self.seg_to_enh_attention(enh_adj, seg_adj)
        
        # 门控融合
        combined = torch.cat([enh_influenced, seg_influenced], dim=1)
        gates = self.gate_net(combined)
        gate_enh, gate_seg = gates.chunk(2, dim=1)
        
        # 加权融合
        fused_enh = enh_influenced * gate_enh + seg_influenced * (1 - gate_enh)
        fused_seg = seg_influenced * gate_seg + enh_influenced * (1 - gate_seg)
        
        # 特征增强
        enh_out = self.enh_enhance(torch.cat([fused_enh, enh_adj], dim=1)) + enh_feat
        seg_out = self.seg_enhance(torch.cat([fused_seg, seg_adj], dim=1)) + seg_feat
        
        return enh_out, seg_out

# ==================== 基于扩散的双分支网络 ====================
class DiEnhBranch(nn.Module):
    """基于扩散的图像增强分支 (DiEnh)"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        # 使用改进的扩散UNet
        self.diffusion_unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim
        )
        
        # 多尺度特征提取器
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
        
        # 质量评估头
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, timesteps, return_features=True):
        # 获取增强图像
        enhanced = self.diffusion_unet(x, timesteps)
        
        if return_features:
            # 提取多尺度特征（这里简化实现）
            B, C, H, W = x.shape
            features = []
            
            # 模拟不同尺度的特征提取
            for i, extractor in enumerate(self.feature_extractors):
                if i == 0:
                    feat = F.interpolate(x, scale_factor=1/(2**i), mode='bilinear')
                else:
                    feat = F.avg_pool2d(features[-1], 2)
                feat = extractor(feat)
                features.append(feat)
            
            # 质量评估
            quality_score = self.quality_head(features[-1])
            
            return enhanced, features, quality_score
        else:
            return enhanced

class DiSegBranch(nn.Module):
    """基于扩散的图像分割分支 (DiSeg)"""
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.n_classes = n_classes
        
        # 扩散分割UNet
        self.diffusion_unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=n_classes,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim
        )
        
        # 边界感知模块
        self.boundary_aware = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(16, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, n_classes, 1)
        )
        
        # 多尺度特征提取器
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
        # 获取分割结果
        segmentation = self.diffusion_unet(x, timesteps)
        
        if return_features:
            # 提取多尺度特征
            B, C, H, W = x.shape
            features = []
            
            # 模拟不同尺度的特征提取
            for i, extractor in enumerate(self.feature_extractors):
                if i == 0:
                    feat = F.interpolate(x, scale_factor=1/(2**i), mode='bilinear')
                else:
                    feat = F.avg_pool2d(features[-1], 2)
                feat = extractor(feat)
                features.append(feat)
            
            # 边界感知输出
            boundary = self.boundary_aware(features[-1])
            boundary = F.interpolate(boundary, size=(H, W), mode='bilinear')
            
            return segmentation, boundary, features
        else:
            return segmentation

# ==================== 完整的DiSIINet ====================
class DiSIINet(nn.Module):
    """基于扩散的共生信息交互网络 (DiSIINet)"""
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, 
                 time_emb_dim=256, num_timesteps=1000):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
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
        
        # 共生信息交互模块 (在不同层级)
        self.sii_modules = nn.ModuleList([
            SymbioticInformationInteraction(
                enh_channels=base_channels * (2**i),
                seg_channels=base_channels * (2**i),
                fusion_channels=base_channels * (2**i)
            )
            for i in range(3)  # 三个交互层级
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
        
        # 初始化beta调度
        self._init_diffusion_schedule()
    
    def _init_diffusion_schedule(self):
        """初始化扩散调度参数"""
        # 线性beta调度
        self.betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        
        # 预计算扩散参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    def forward(self, noisy_image, timesteps, return_all=False):
        """
        Args:
            noisy_image: 带噪图像 [B, C, H, W]
            timesteps: 时间步 [B]
            return_all: 是否返回所有中间结果
        """
        B = noisy_image.shape[0]
        
        # 编码时间步
        t_emb = self.timestep_encoder(timesteps)
        
        # 双分支前向传播（初始）
        enhanced_init, enh_features_init, quality_score = self.enh_branch(noisy_image, timesteps)
        segmented_init, boundary_init, seg_features_init = self.seg_branch(noisy_image, timesteps)
        
        # 多层级共生信息交互
        enh_features = []
        seg_features = []
        
        for i, sii_module in enumerate(self.sii_modules):
            # 获取对应层级的特征
            enh_feat = enh_features_init[i] if i < len(enh_features_init) else enh_features[-1]
            seg_feat = seg_features_init[i] if i < len(seg_features_init) else seg_features[-1]
            
            # 共生信息交互
            enh_feat_interacted, seg_feat_interacted = sii_module(enh_feat, seg_feat)
            
            enh_features.append(enh_feat_interacted)
            seg_features.append(seg_feat_interacted)
        
        # 使用交互后的特征进行最终预测
        # 合并最高层特征进行联合优化
        combined_feat = torch.cat([enh_features[-1], seg_features[-1]], dim=1)
        joint_output = self.joint_optimizer(combined_feat)
        
        # 最终输出
        enhanced_final = enhanced_init + joint_output  # 增强图像
        segmented_final = segmented_init + F.interpolate(joint_output, size=segmented_init.shape[2:])  # 分割结果
        
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
    
    def sample_timesteps(self, batch_size, device):
        """采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, seg_gt, t, noise=None):
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 添加噪声
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 网络预测
        enhanced, segmented = self(x_noisy, t)
        
        # 重建损失
        recon_loss = F.mse_loss(enhanced, x_start)
        
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
    
    @torch.no_grad()
    def sample(self, image, num_samples=1, return_intermediates=False):
        """采样生成增强图像和分割结果"""
        device = image.device
        b = image.shape[0]
        
        # 初始化噪声
        img = torch.randn_like(image)
        
        intermediates = []
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # 预测
            enhanced, segmented = self(img, t)
            
            if return_intermediates and i % 50 == 0:
                intermediates.append((enhanced, segmented))
            
            # 去噪（简化的DDIM采样）
            if i > 0:
                noise = torch.randn_like(img)
                beta_t = self.betas[t].reshape(-1, 1, 1, 1)
                alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
                alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
                alpha_cumprod_t_prev = self.alphas_cumprod_prev[t].reshape(-1, 1, 1, 1)
                
                # DDIM采样公式
                pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * enhanced) / torch.sqrt(alpha_cumprod_t)
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * enhanced
                img = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
        
        if return_intermediates:
            return enhanced, segmented, intermediates
        return enhanced, segmented

# ==================== 训练和损失函数 ====================
class DiSIINetLoss(nn.Module):
    """DiSIINet的损失函数"""
    def __init__(self, lambda_recon=1.0, lambda_seg=1.0, lambda_boundary=0.5, lambda_quality=0.1):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_seg = lambda_seg
        self.lambda_boundary = lambda_boundary
        self.lambda_quality = lambda_quality
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 网络输出字典
            targets: 包含clean_image和segmentation_gt的字典
        """
        clean_image = targets['clean_image']
        seg_gt = targets['segmentation_gt']
        
        # 重建损失
        recon_loss = F.mse_loss(predictions['enhanced'], clean_image)
        
        # 分割损失
        seg_loss = F.binary_cross_entropy_with_logits(predictions['segmented'], seg_gt)
        
        # 边界损失（如果存在）
        if 'boundary' in predictions:
            # 从分割GT计算边界
            seg_gt_boundary = self._compute_boundary(seg_gt)
            boundary_loss = F.mse_loss(predictions['boundary'], seg_gt_boundary)
        else:
            boundary_loss = torch.tensor(0.0, device=clean_image.device)
        
        # 质量一致性损失
        if 'quality_score' in predictions:
            # 假设质量分数应该与分割精度正相关
            seg_accuracy = self._compute_seg_accuracy(predictions['segmented'], seg_gt)
            quality_loss = F.mse_loss(predictions['quality_score'], seg_accuracy.unsqueeze(1))
        else:
            quality_loss = torch.tensor(0.0, device=clean_image.device)
        
        # 共生一致性损失
        sym_loss = self._compute_symbiotic_loss(predictions)
        
        # 总损失
        total_loss = (self.lambda_recon * recon_loss +
                     self.lambda_seg * seg_loss +
                     self.lambda_boundary * boundary_loss +
                     self.lambda_quality * quality_loss +
                     0.1 * sym_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'seg_loss': seg_loss,
            'boundary_loss': boundary_loss,
            'quality_loss': quality_loss,
            'sym_loss': sym_loss
        }
    
    def _compute_boundary(self, seg_map):
        """计算分割图的边界"""
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                             dtype=torch.float32, device=seg_map.device).view(1, 1, 3, 3)
        boundary = F.conv2d(seg_map, kernel, padding=1)
        boundary = torch.abs(boundary)
        boundary = torch.clamp(boundary, 0, 1)
        return boundary
    
    def _compute_seg_accuracy(self, pred, gt):
        """计算分割准确率"""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        correct = (pred_binary == gt).float()
        accuracy = correct.mean(dim=(1, 2, 3))
        return accuracy
    
    def _compute_symbiotic_loss(self, predictions):
        """计算共生一致性损失"""
        if 'enhanced_init' in predictions and 'segmented_init' in predictions:
            # 初始预测和交互后预测应该一致
            enhanced_consistency = F.mse_loss(predictions['enhanced'], predictions['enhanced_init'])
            segmented_consistency = F.mse_loss(predictions['segmented'], predictions['segmented_init'])
            return enhanced_consistency + segmented_consistency
        return torch.tensor(0.0, device=next(iter(predictions.values())).device)

# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 DiSIINet 网络")
    print("=" * 60)
    
    # 创建模型
    model = DiSIINet(
        in_channels=3,
        n_classes=1,
        base_channels=32,  # 为了测试使用较小的通道数
        time_emb_dim=128,
        num_timesteps=100
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
    
    # 计算损失
    targets = {
        'clean_image': torch.randn_like(image),
        'segmentation_gt': seg_gt
    }
    
    predictions = {
        'enhanced': enhanced,
        'segmented': segmented,
        'boundary': torch.randn(batch_size, 1, 128, 128),
        'quality_score': torch.randn(batch_size, 1)
    }
    
    loss_fn = DiSIINetLoss()
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
    
    # 测试采样
    print("\n测试采样过程...")
    enhanced_sample, segmented_sample = model.sample(image)
    print(f"采样增强图像形状: {enhanced_sample.shape}")
    print(f"采样分割结果形状: {segmented_sample.shape}")
    
    # 测试损失计算
    print("\n测试损失计算...")
    loss_dict = model.p_losses(image, seg_gt, timesteps)
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")
    
    print("\nDiSIINet测试完成!")

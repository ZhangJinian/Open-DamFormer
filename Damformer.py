import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# 1. 首先实现Mix Transformer的基本组件
class EfficientSelfAttention(nn.Module):
    """高效自注意力机制，用于Mix Transformer块"""
    def __init__(self, dim, num_heads=4, qkv_bias=False, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    """Mix Feed-Forward Network"""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """Mix Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class OverlapPatchEmbed(nn.Module):
    """重叠patch嵌入"""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class MiT(nn.Module):
    """Mix Transformer encoder (简化版SegFormer编码器)"""
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=256, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=128, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=64, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=32, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        # transformer blocks
        cur = 0
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            sr_ratio=sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        
    def forward(self, x):
        B = x.shape[0]
        outs = []
        
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        return outs

# 2. 多时相自适应融合模块
class ChannelAttention(nn.Module):
    """通道注意力机制 (CBAM)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class TemporalFusionModule(nn.Module):
    """多时相自适应融合模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_attn = ChannelAttention(in_channels)
        
    def forward(self, x1, x2):
        # x1, x2: [B, C, H, W]
        x_concat = torch.cat([x1, x2], dim=1)
        x_fused = self.conv(x_concat)
        x_out = self.channel_attn(x_fused)
        return x_out

# 3. 轻量级双任务解码器
class MLPDecoder(nn.Module):
    """轻量级MLP解码器"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class DualTasksDecoder(nn.Module):
    """双任务解码器"""
    def __init__(self, feature_channels=[64, 128, 256, 512], num_classes=4):
        super().__init__()
        # 建筑定位分支
        self.building_decoder = nn.ModuleList()
        # 损伤分类分支
        self.damage_decoder = nn.ModuleList()
        
        # 为每个层级创建解码器
        for i in range(len(feature_channels)):
            # 建筑定位分支
            self.building_decoder.append(
                MLPDecoder(feature_channels[i], 256)
            )
            # 损伤分类分支
            self.damage_decoder.append(
                MLPDecoder(feature_channels[i], 256)
            )
        
        # 跨层融合
        self.building_fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.damage_fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 最终分类器
        self.building_classifier = nn.Conv2d(256, 1, kernel_size=1)
        self.damage_classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, building_features, damage_features):
        B, _, H, W = building_features[0].shape
        
        # 建筑定位分支
        building_outputs = []
        for i, feat in enumerate(building_features):
            out = self.building_decoder[i](feat)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            building_outputs.append(out)
            
        building_fused = torch.cat(building_outputs, dim=1)
        building_fused = self.building_fusion(building_fused)
        building_pred = self.building_classifier(building_fused)
        
        # 损伤分类分支
        damage_outputs = []
        for i, feat in enumerate(damage_features):
               
            out = self.damage_decoder[i](feat)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            damage_outputs.append(out)
            
        damage_fused = torch.cat(damage_outputs, dim=1)
        damage_fused = self.damage_fusion(damage_fused)
        final_feat = damage_fused + building_fused
        damage_pred = self.damage_classifier(final_feat)
        
        return building_pred, damage_pred

# 4. 完整的DamFormer架构
class DamFormer(nn.Module):
    """完整的DamFormer架构"""
    def __init__(self, in_channels=3, num_classes=4, 
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 depths=[3, 4, 6, 3]):
        super().__init__()
        # 孪生Transformer编码器
        self.encoder = MiT(
            in_chans=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=[8, 4, 2, 1]
        )
        
        # 多时相融合模块 (4个层级)
        self.fusion_modules = nn.ModuleList([
            TemporalFusionModule(embed_dims[i]) for i in range(len(embed_dims))
        ])
        
        # 双任务解码器
        self.decoder = DualTasksDecoder(
            feature_channels=embed_dims,
            num_classes=num_classes
        )
        
    def forward(self, pre_disaster, post_disaster):
        # 编码器提取特征
        pre_features = self.encoder(pre_disaster)
        post_features = self.encoder(post_disaster)
        
        # 多时相融合
        fused_features = []
        for i in range(len(pre_features)):
            fused_feat = self.fusion_modules[i](pre_features[i], post_features[i])
            fused_features.append(fused_feat)
        
        # 双任务解码
        building_pred, damage_pred = self.decoder(fused_features, fused_features)
        
        # 尺寸调整到输入大小
        building_pred = F.interpolate(
            building_pred, size=pre_disaster.shape[2:], mode='bilinear', align_corners=False
        )
        damage_pred = F.interpolate(
            damage_pred, size=pre_disaster.shape[2:], mode='bilinear', align_corners=False
        )
        
        return {
            'building': torch.sigmoid(building_pred),
            'damage': damage_pred
        }


# 4. 完整的DamFormer架构
class MyDamFormer(nn.Module):
    """完整的DamFormer架构"""
    def __init__(self, in_channels=3, num_classes=4, 
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 depths=[3, 4, 6, 3]):
        super().__init__()
        # 孪生Transformer编码器
        self.encoder1 = MiT(
            in_chans=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=[8, 4, 2, 1]
        )
        self.encoder2 = MiT(
            in_chans=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=[8, 4, 2, 1]
        )
        
        # 多时相融合模块 (4个层级)
        self.fusion_modules = nn.ModuleList([
            TemporalFusionModule(embed_dims[i]) for i in range(len(embed_dims))
        ])
        
        # 双任务解码器
        self.decoder = DualTasksDecoder(
            feature_channels=embed_dims,
            num_classes=num_classes
        )
        
    def forward(self, pre_disaster, post_disaster):
        # 编码器提取特征
        pre_features = self.encoder1(pre_disaster)
        post_features = self.encoder2(post_disaster)
        
        # 多时相融合
        fused_features = []
        for i in range(len(pre_features)):
            fused_feat = self.fusion_modules[i](pre_features[i], post_features[i])
            fused_features.append(fused_feat)
        
        # 双任务解码
        building_pred, damage_pred = self.decoder(fused_features, fused_features)
        
        # 尺寸调整到输入大小
        building_pred = F.interpolate(
            building_pred, size=pre_disaster.shape[2:], mode='bilinear', align_corners=False
        )
        damage_pred = F.interpolate(
            damage_pred, size=pre_disaster.shape[2:], mode='bilinear', align_corners=False
        )
        
        return {
            'building': torch.sigmoid(building_pred),
            'damage': damage_pred
        }

# 测试DamFormer
# if __name__ == "__main__":
#     # 创建模型实例
#     model = DamFormer(in_channels=3, num_classes=4)
    
#     # 模拟输入 (batch_size=2, channels=3, height=256, width=256)
#     pre_img = torch.randn(2, 3, 256, 256)
#     post_img = torch.randn(2, 3, 256, 256)
    
#     # 前向传播
#     outputs = model(pre_img, post_img)
    
#     print("Building prediction shape:", outputs['building'].shape)
#     print("Damage prediction shape:", outputs['damage'].shape)
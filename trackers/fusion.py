# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmdet.registry import MODELS


# Channel-Spatial Attention
class ChannelSpatialAttention(nn.Module):
    """
    Lightweight Channel-Spatial Attention for Template Feature Enhancement.
    """

    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()

        # 1. Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        # 2. Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.channel_sigmoid(avg_out + max_out)
        x = x * channel_attn  # Channel-enhanced feature

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x_in = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_sigmoid(self.spatial_conv(x_in))

        return x * spatial_attn  # Spatially-enhanced feature


# Single-Scale Cross-Attention Block


class CrossAttentionBlock(nn.Module):
    """
    Template-to-Search Cross-Attention Block for a single feature level.
    S_feat (Search) is Query, T_feat (Template) is Key/Value.
    """

    def __init__(self, in_channels: int, norm_cfg=dict(type="BN", requires_grad=True)):
        super().__init__()

        # QKV Projections (using 1x1 Conv for efficiency in FPN features)
        self.q_proj = ConvModule(
            in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None
        )  # Search (Query)
        self.k_proj = ConvModule(
            in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None
        )  # Template (Key)
        self.v_proj = ConvModule(
            in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None
        )  # Template (Value)

        # Output Projection (Feed Forward Network equivalent)
        self.ffn = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU"),
            ),
            ConvModule(in_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None),
        )
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.scale = in_channels**-0.5

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        B, C, H_s, W_s = s_feat.shape

        # 1. Project Q, K, V
        Q = self.q_proj(s_feat).flatten(2).permute(0, 2, 1)  # B, H_s*W_s, C
        K = self.k_proj(t_feat).flatten(2).permute(0, 2, 1)  # B, H_t*W_t, C
        V = self.v_proj(t_feat).flatten(2).permute(0, 2, 1)  # B, H_t*W_t, C

        # 2. Attention: (Q * K^T) / scale
        attn_weights = (Q @ K.transpose(-2, -1)) * self.scale  # B, H_s*W_s, H_t*W_t
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 3. Output: Attn * V
        output = attn_weights @ V  # B, H_s*W_s, C
        output = output.permute(0, 2, 1).view(B, C, H_s, W_s)  # B, C, H_s, W_s

        # 4. Residual Connection + FFN
        fused_feat = s_feat + output
        fused_feat = self.norm(fused_feat)
        fused_feat = fused_feat + self.ffn(fused_feat)

        return fused_feat


# Multi-Scale Feature Despeckling and Cross-Attention, MFDCAFusion
@MODELS.register_module()
class MFDCAFusion(nn.Module):
    """
    Multi-Scale Feature Despeckling and Cross-Attention Fusion.
    Handles multi-scale FPN features (Tuple[Tensor]).
    """

    def __init__(
        self, in_channels: List[int], norm_cfg=dict(type="BN", requires_grad=True)
    ):
        super().__init__()

        self.num_levels = len(in_channels)
        self.template_enhancers = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()

        for i in range(self.num_levels):
            channels = in_channels[i]

            # 模板增强/去噪模块（仅应用于模板特征T）
            self.template_enhancers.append(ChannelSpatialAttention(channels))

            # T -> S 引导
            self.cross_attentions.append(
                CrossAttentionBlock(channels, norm_cfg=norm_cfg)
            )

    def forward(
        self, t_feats: Tuple[torch.Tensor], s_feats: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            t_feats (Tuple[Tensor]): Template multi-scale features (e.g., P2, P3, P4, P5).
            s_feats (Tuple[Tensor]): Search multi-scale features (e.g., P2, P3, P4, P5).

        Returns:
            Tuple[Tensor]: Fused multi-scale features, matching the S_feats scale.
        """
        assert len(t_feats) == len(s_feats) == self.num_levels
        fused_feats = []

        # 逐层进行融合
        for i in range(self.num_levels):
            t_feat = t_feats[i]
            s_feat = s_feats[i]

            # 模板特征增强/去噪
            t_feat_enhanced = self.template_enhancers[i](t_feat)

            # 融合 (T_enhanced -> S)
            fused_feat = self.cross_attentions[i](t_feat_enhanced, s_feat)

            fused_feats.append(fused_feat)

        return tuple(fused_feats)

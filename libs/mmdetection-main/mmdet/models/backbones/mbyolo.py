import torch
import math
from functools import partial
from typing import Callable, Any, List, Dict, Optional,Sequence,Tuple

import torch.nn as nn
import torch.utils.checkpoint as cp  # 添加这一行导入cp
from einops import rearrange, repeat
from timm.models.layers import DropPath

from mmengine.model import BaseModule
from mmdet.registry import MODELS

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    # import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    # sscore acts the same as mamba_ssm
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    # you should install mamba_ssm to use this
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# 添加CrossScan类定义
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


# 添加CrossMerge类定义
class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default'
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ==============
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv =================
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ========================
            dropout=0.0,
            bias=False,
            # ========================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        # --------------------------
        # 基础参数计算
        # --------------------------
        d_expand = max(1, int(ssm_ratio * d_model))   # ✅ 防止出现 0
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4
        self.d_expand = d_expand
        self.d_inner = d_inner

        # --------------------------
        # forward_type 参数解析
        # --------------------------
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # --------------------------
        # out_norm 修正 ✅
        # --------------------------
        self.out_norm = nn.LayerNorm(self.d_inner)  # 保证最后一维对齐

        # forward_type 映射
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # --------------------------
        # in_proj 输入映射
        # --------------------------
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # --------------------------
        # depthwise conv ✅ (防止 groups=0)
        # --------------------------
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,   # ✅ 修复 groups 错误
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # --------------------------
        # rank 降维通道映射
        # --------------------------
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # ✅ 添加自动对齐层（修正通道不匹配）
        if self.d_expand != self.d_inner:
            self.align_proj = nn.Conv2d(self.d_expand, self.d_inner, kernel_size=1)
        else:
            self.align_proj = nn.Identity()

        # --------------------------
        # 线性投影参数
        # --------------------------
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # --------------------------
        # 输出投影与参数初始化
        # --------------------------
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    # ============================================================
    # forward core
    # ============================================================
    def forward_corev2(self, x, **kwargs):
        """
        x: [B, C, H, W]
        """
        # ✅ 对齐输入通道
        x = self.align_proj(x)
    
        # 从kwargs中移除不支持的参数
        if 'SelectiveScan' in kwargs:
            del kwargs['SelectiveScan']
        
        # 移除channel_first参数，因为cross_selective_scan不支持这个参数
        if 'channel_first' in kwargs:
            del kwargs['channel_first']
            
        # 其他原始逻辑保持不变
        return cross_selective_scan(
            x=x,
            x_proj_weight=self.x_proj_weight,
            x_proj_bias=None,
            dt_projs_weight=self.dt_projs_weight,
            dt_projs_bias=self.dt_projs_bias,
            A_logs=self.A_logs,
            Ds=self.Ds,
            out_norm=self.out_norm,
            SelectiveScan=SelectiveScanCore,
            **kwargs
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        if hidden_features > 0:
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                    groups=hidden_features)
        else:
            # 处理 hidden_features 为 0 的情况，比如替换为普通卷积（groups=1）
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                    groups=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_features,
            hidden_features,
            kernel_size=3,
            padding=3 // 2,
            groups=int(max(1, round(hidden_features)))  # 强制为正整数
        )

        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = input + x
        return x


class MambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop=0.,
            forward_type="v2",
            **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.op = SS2D(
            d_model=dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop,
            forward_type=forward_type,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = RGBlock(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
        # 检查是否为标准LayerNorm，如果是则需要特殊处理
        self.is_standard_ln = isinstance(norm_layer, type) and norm_layer == nn.LayerNorm

    def forward(self, x):
        if self.is_standard_ln:
            # 对于标准LayerNorm，需要调整维度顺序
            B, C, H, W = x.shape
            x_norm1 = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, H*W, C]
            x_norm1 = self.norm1(x_norm1).reshape(B, H, W, C).permute(0, 3, 1, 2)  # 回到[B, C, H, W]
            x = x + self.drop_path(self.op(x_norm1))
            
            x_norm2 = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            x_norm2 = self.norm2(x_norm2).reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = x + self.drop_path(self.mlp(x_norm2))
        else:
            # 对于LayerNorm2d等已适配4D张量的归一化层，直接使用
            x = x + self.drop_path(self.op(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MambaStage(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop=0.,
            downsample=None,
            use_checkpoint=False,
            forward_type="v2",
            **kwargs
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act=ssm_act,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop=ssm_drop,
                forward_type=forward_type,
                **kwargs
            )
            for i in range(depth)])
        self.downsample = downsample
        self.dim = dim

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.permute(0, 2, 3, 1).contiguous()

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        return x
class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input
class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)

class Conv1x1BN(nn.Module):
    """helper to align channels if needed"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
# 在MambaVision类的__init__方法中，确保使用LayerNorm2d
# @MODELS.register_module()
# class MambaVision(BaseModule):
#     def __init__(self,
#                  in_chans=3,
#                  embed_dims=[96, 192, 384, 768],
#                  depths=[2, 2, 9, 2],
#                  mlp_ratio=4.,
#                  drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_layer=LayerNorm2d,  # 使用LayerNorm2d作为默认值
#                  patch_norm=True,
#                  out_indices=(0, 1, 2, 3),
#                  frozen_stages=-1,
#                  use_checkpoint=False,
#                  init_cfg=None,
#                  ssm_d_state=16,
#                  ssm_ratio=2.0,
#                  ssm_rank_ratio=2.0,
#                  ssm_dt_rank="auto",
#                  ssm_act=nn.SiLU,
#                  ssm_conv=3,
#                  ssm_conv_bias=True,
#                  ssm_drop=0.,
#                  forward_type="v2",
#                  **kwargs):
#         super().__init__(init_cfg=init_cfg)
#         self.num_layers = len(depths)
#         self.embed_dims = embed_dims
#         self.patch_norm = patch_norm
#         self.out_indices = out_indices
#         self.frozen_stages = frozen_stages
        
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             in_chans=in_chans, embed_dim=embed_dims[0], patch_size=4,
#             norm_layer=norm_layer if self.patch_norm else None)
        
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = MambaStage(
#                 dim=embed_dims[i_layer],
#                 depth=depths[i_layer],
#                 mlp_ratio=mlp_ratio,
#                 drop=drop_rate,
#                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                 norm_layer=norm_layer,
#                 downsample=PatchMerging(
#                     dim=embed_dims[i_layer],
#                     out_dim=embed_dims[i_layer + 1] if i_layer < self.num_layers - 1 else None,
#                     norm_layer=norm_layer
#                 ) if i_layer < self.num_layers - 1 else None,
#                 use_checkpoint=use_checkpoint,
#                 ssm_d_state=ssm_d_state,
#                 ssm_ratio=ssm_ratio,
#                 ssm_rank_ratio=ssm_rank_ratio,
#                 ssm_dt_rank=ssm_dt_rank,
#                 ssm_act=ssm_act,
#                 ssm_conv=ssm_conv,
#                 ssm_conv_bias=ssm_conv_bias,
#                 ssm_drop=ssm_drop,
#                 forward_type=forward_type,
#                 **kwargs
#             )
#             self.layers.append(layer)
        
#         # add a norm layer for each output
#         for i_layer in out_indices:
#             layer = norm_layer(embed_dims[i_layer])
#             layer_name = f'norm{i_layer}'
#             self.add_module(layer_name, layer)
        
#         self._freeze_stages()
    
#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.patch_embed.eval()
#             for param in self.patch_embed.parameters():
#                 param.requires_grad = False
        
#         for i in range(0, self.frozen_stages):
#             m = self.layers[i]
#             m.eval()
#             for param in m.parameters():
#                 param.requires_grad = False
    
#     def forward(self, x):
#         x = self.patch_embed(x)
        
#         outs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i in self.out_indices:
#                 norm_layer = getattr(self, f'norm{i}')
#                 out = norm_layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#                 outs.append(out)
        
        # return tuple(outs)
        
@MODELS.register_module()
class MambaVision(BaseModule):
    """
    A wrapper that builds a stage-based backbone using Mamba blocks and registers to MMDetection.
    This implementation is intentionally generic — tune `depths`, `embed_dims`, `hidden_dims`
    to match the exact Mamba-YOLO architecture you want.

    Notes:
      - `SimpleStem`, `VSSBlock`, `XSSBlock`, `VisionClueMerge` are expected to be available
        (importable from ultralytics.nn.mamba_yolo or local copy).
      - `out_indices` chooses which stages are returned to the neck (0-based).
    """

    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: Sequence[int] = (96, 192, 384, 768),
                 depths: Sequence[int] = (2, 2, 6, 2),
                 block_type: str = "VSS",  # "VSS" or "XSS"
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 align_out_channels: Optional[Sequence[int]] = None,
                 init_cfg=None):
        """
        embed_dims: channels for each stage output
        depths: number of blocks per stage
        block_type: which block to use (VSSBlock or XSSBlock)
        out_indices: which stage outputs to return
        align_out_channels: if provided, 1x1 convs will map stage channels to these channels
        """
        super().__init__(init_cfg=init_cfg)
        assert len(embed_dims) == len(depths)
        self.in_channels = in_channels
        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.num_stages = len(self.embed_dims)
        self.out_indices = tuple(out_indices)
        self.block_type = block_type.upper()
        self.align_out_channels = list(align_out_channels) if align_out_channels is not None else None

        # Stem
        self.stem = SimpleStem(inp=in_channels, embed_dim=self.embed_dims[0], ks=3)

        # Build stages
        self.stages = nn.ModuleList()
        # optionally a merging block between stages (e.g., VisionClueMerge) to downsample / merge
        self.merges = nn.ModuleList()

        for i in range(self.num_stages):
            ch = self.embed_dims[i]
            depth = self.depths[i]
            blocks = []
            # choose block class
            BlockClass = VSSBlock if self.block_type == "VSS" else XSSBlock
            # first block of stage expects current channel, others use same channel
            for j in range(depth):
                in_ch = ch if j > 0 else ch
                # default args here, you can expose more in init if desired
                blocks.append(BlockClass(in_channels=in_ch, hidden_dim=ch))
            self.stages.append(nn.Sequential(*blocks))
            # If not last stage, add a VisionClueMerge to downsample spatially (merging 2x2 -> reduce H/2 W/2)
            if i < self.num_stages - 1:
                # merge from current stage channels to next stage channels
                out_ch = self.embed_dims[i + 1]
                self.merges.append(VisionClueMerge(dim=ch, out_dim=out_ch))
        # if align_out_channels provided, add 1x1 convs
        if self.align_out_channels is not None:
            assert len(self.align_out_channels) == self.num_stages
            self.align_convs = nn.ModuleList([
                Conv1x1BN(self.embed_dims[i], self.align_out_channels[i]) for i in range(self.num_stages)
            ])
            self._out_channels = tuple(self.align_out_channels)
        else:
            self.align_convs = None
            self._out_channels = tuple(self.embed_dims)

    @property
    def out_channels(self) -> Tuple[int, ...]:
        """Return tuple of channels for each stage (useful for some mm code paths)."""
        return self._out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns tuple of stage feature maps (ordered from stage0 ... stageN-1)
        Shapes: spatial size reduced progressively (depending on stem / merges).
        """
        outs: List[torch.Tensor] = []
        # stem
        x = self.stem(x)  # downsampled 2x twice inside SimpleStem (so /4) — check your stem impl
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # add output
            out_i = x
            if self.align_convs is not None:
                out_i = self.align_convs[i](out_i)
            outs.append(out_i)
            # merge/downsample to next stage if exists
            if i < len(self.merges):
                x = self.merges[i](x)  # VisionClueMerge does 2x2 concat -> conv, so reduces spatial by 2
        # select indices
        selected = tuple(outs[i] for i in self.out_indices)
        return selected

    def init_weights(self, pretrained: Optional[str] = None, **kwargs):
        """
        If pretrained path provided, try loading with strict=False; otherwise default init.
        Note: when using MMDetection's init_cfg mechanism this might not be called with a path.
        """
        if pretrained:
            state = torch.load(pretrained, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            try:
                self.load_state_dict(state, strict=False)
                print(f"[MambaBackbone] Loaded pretrained weights from {pretrained} (strict=False).")
                return
            except Exception as e:
                print(f"[MambaBackbone] Warning: failed to load pretrained strictly: {e}. Trying partial load.")
                try:
                    self.load_state_dict(state, strict=False)
                except Exception as e2:
                    print(f"[MambaBackbone] Partial load failed: {e2}. Proceeding with default init.")
        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keeping normalization layers
    #     frozen."""
    #     super().train(mode)
    #     self._freeze_stages()


from typing import Tuple, List, Optional
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader
from functools import partial
from torch import Tensor


try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.utils.generation import GenerationMixin
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
except ImportError:
    Mamba = None
    GenerationMixin = None
    load_config_hf = None
    load_state_dict_hf = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
##
try:
    from mamba_ssm.modules.mamba_simple import Mamba as ExternalMamba
except Exception:
    ExternalMamba = None
class MambaWrapper(nn.Module):
    """
    Accepts bimamba_type but does not require underlying ExternalMamba to accept it.
    If ExternalMamba exists and can be constructed, delegate; otherwise use a small fallback core.
    """
    def __init__(self, dim, d_state=16, layer_idx=None,
                 bimamba_type=None, if_divide_out=False, init_layer_scale=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out

        self._use_external = False
        if ExternalMamba is not None:
            # 尝试用安全的方式构造 ExternalMamba（不传 bimamba_type）
            filtered = dict(kwargs)
            filtered.pop('bimamba_type', None)
            try:
                # 常见签名: ExternalMamba(dim, d_state=..., layer_idx=..., ...)
                self.inner = ExternalMamba(dim, d_state=d_state, layer_idx=layer_idx,
                                           if_divide_out=if_divide_out,
                                           init_layer_scale=init_layer_scale,
                                           **filtered)
                self._use_external = True
            except TypeError:
                try:
                    # 退一步尝试最简单的位置参数签名
                    self.inner = ExternalMamba(dim, d_state, layer_idx)
                    self._use_external = True
                except Exception:
                    self._use_external = False

        if not self._use_external:
            # fallback: minimal implementation depending on bimamba_type
            if bimamba_type == "v1":
                self.core = nn.Linear(dim, dim)
            elif bimamba_type == "v2":
                self.core = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            else:
                self.core = nn.Identity()

    def forward(self, x, inference_params=None):
        if self._use_external:
            return self.inner(x, inference_params=inference_params)
        out = self.core(x)
        if self.if_divide_out:
            out = out / (self.d_state ** 0.5)
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if self._use_external and hasattr(self.inner, 'allocate_inference_cache'):
            return self.inner.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return None
    

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)  # 确保 stride 也是二元组
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride  # 在使用之前就定义好
        self.flatten = flatten
        
        # 使用 stride 计算 grid_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride[0] + 1, 
                         (img_size[1] - patch_size[1]) // stride[1] + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 使用 stride 元组的第一个元素作为卷积的 stride
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride[0])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _get_num_patches(self, H, W):
        """动态计算实际输入尺寸的 num_patches"""
        grid_h = (H - self.patch_size[0]) // self.stride[0] + 1
        grid_w = (W - self.patch_size[1]) // self.stride[1] + 1
        return grid_h * grid_w

    def forward(self, x):
        B, C, H, W = x.shape
        # 更新 num_patches 为实际值
        self.num_patches = self._get_num_patches(H, W)
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        d_state,
        mixer_cls: nn.Module,            # 例如Mamba或其它
        norm_cls=nn.LayerNorm, 
       
        fused_add_norm=False, 
        residual_in_fp32=False, 
        drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
   
      
        # 实例化 mixer 时传 bimamba_type
        self.mixer = mixer_cls( )
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    MambaWrapper,   # 用包装好的类
    d_model,
    d_state,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    if Mamba is None:
        raise ImportError("mamba_ssm is required for VMamba. Please install it with: pip install mamba-ssm")
    

    mixer_cls = partial(
        MambaWrapper,
        dim=d_model,
        d_state=d_state,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        d_state,
        mixer_cls= mixer_cls,
       
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



@MODELS.register_module()
class VMamba(BaseModule):
    """
    VMamba backbone based on official Vim implementation.
    
    This is a simplified version that outputs multi-scale features for FPN.
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        d_state=16,
        channels=3,
        ssm_cfg=None,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon=1e-5,
        rms_norm=True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        ft_seq_len=None,
        pt_hw_seq_len=14,
        if_bidirectional=False,
        final_pool_type='none',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.,
        if_bimamba=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        init_layer_scale=None,
        use_double_cls_token=False,
        use_middle_cls_token=True,
        out_indices=(1, 2, 3),
        init_cfg=None,
        **kwargs
    ):
        super().__init__(init_cfg)
        
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.out_indices = out_indices

        self.d_model = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        # Create layers with different depths for multi-scale output
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Create layers
        self.layers = nn.ModuleList([
            create_block(                
                MambaWrapper,
                embed_dim,
                d_state=d_state,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_divide_out=if_divide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])
        
        # Output norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # Initialize weights
        self.patch_embed.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    def interpolate_pos_encoding(self, x, pos_embed):
        """动态调整位置编码以匹配输入特征的尺寸"""
        batch_size, seq_len, embed_dim = x.shape
        num_extra_tokens = self.num_tokens
        
        # 分离额外的 token（如 cls token）和位置编码
        if num_extra_tokens > 0:
            pos_emb_tok, pos_emb_grid = pos_embed[:, :num_extra_tokens], pos_embed[:, num_extra_tokens:]
        else:
            pos_emb_tok, pos_emb_grid = None, pos_embed

        # 计算当前输入的 grid 尺寸
        grid_size = int(math.sqrt(seq_len - num_extra_tokens))
        
        # 计算原始位置编码的 grid 尺寸
        orig_grid_size = int(math.sqrt(pos_emb_grid.shape[1]))

        # 如果尺寸不同，进行插值
        if orig_grid_size != grid_size:
            pos_emb_grid = pos_emb_grid.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)
            pos_emb_grid = F.interpolate(
                pos_emb_grid, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
            pos_emb_grid = pos_emb_grid.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)
            
            # 重新组合位置编码
            if num_extra_tokens > 0:
                pos_embed = torch.cat((pos_emb_tok, pos_emb_grid), dim=1)
            else:
                pos_embed = pos_emb_grid

        return pos_embed
    
    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(batch_size, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(batch_size, -1, -1)
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(batch_size, -1, -1)
                    token_position = seq_len // 2
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                else:
                    cls_token = self.cls_token.expand(batch_size, -1, -1)
                    x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            # 动态调整位置编码以匹配当前输入
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
            x = x + pos_embed
            x = self.pos_drop(x)

        # Mamba layers
        residual = None
        hidden_states = x
        outs = []
        
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            
            # Collect outputs at specified layers
            if i in self.out_indices:
                # Remove cls token if present
                if self.if_cls_token:
                    if self.use_double_cls_token:
                        # Remove head and tail cls tokens
                        feat = hidden_states[:, 1:-1, :]
                    else:
                        if self.use_middle_cls_token:
                            # Remove middle cls token
                            feat = torch.cat([hidden_states[:, :token_position, :], 
                                           hidden_states[:, token_position+1:, :]], dim=1)
                        else:
                            # Remove first cls token
                            feat = hidden_states[:, 1:, :]
                else:
                    feat = hidden_states
                
                # Reshape to spatial format
                batch_size, num_tokens, channels = feat.shape
                height = width = int(math.sqrt(num_tokens))
                feat = feat.transpose(1, 2).reshape(batch_size, channels, height, width)
                outs.append(feat)

        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        return tuple(outs)

    def init_weights(self):
        """Initialize weights from pretrained checkpoint."""
        logger = MMLogger.get_current_instance()
        
        if self.init_cfg is None or 'checkpoint' not in self.init_cfg:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch')
            return
        
        ckpt = CheckpointLoader.load_checkpoint(self.init_cfg.checkpoint, logger=logger, map_location='cpu')
        
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        # Strip common prefixes
        cleaned = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('backbone.'):
                name = name[len('backbone.'):]
            if name.startswith('module.'):
                name = name[len('module.'):]
            cleaned[name] = v
        
        # Load with strict=False to handle missing/unexpected keys
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning(f'{self.__class__.__name__}: missing keys when loading pretrained: {len(missing)} (showing first 10) {missing[:10]}')
        if unexpected:
            logger.warning(f'{self.__class__.__name__}: unexpected keys when loading pretrained: {len(unexpected)} (showing first 10) {unexpected[:10]}')



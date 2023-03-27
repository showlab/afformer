import torch
from torch import nn
from typing import Tuple
from functools import partial

from timm.models.layers import trunc_normal_, Mlp, DropPath
import xformers.ops as xops
from torchvision.models.video.mvit import _interpolate

def get_rel_pos(q_size, k_size, rel_pos, pad=True):
    q_ratio = max(k_size / q_size, 1.0)
    k_ratio = max(q_size / k_size, 1.0)
    dist = torch.arange(q_size)[:, None] * q_ratio - (
        torch.arange(k_size)[None, :] + (1.0 - k_size)
    ) * k_ratio
    if pad:
        d = int(2 * max(q_size, k_size) - 1)
        rel_pos = _interpolate(rel_pos, d)
    return rel_pos[dist.long()]

# modified from detectron2.modeling.backbone.utils.add_decomposed_rel_pos
def get_decomposed_rel_pos_2d(q, rel_pos_h, rel_pos_w, q_size, k_size, num_heads):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    q = q.unflatten(1, (q_h, q_w))
    rel_h = torch.einsum("bhwnc,hkc->bnhwk", q, Rh)
    rel_w = torch.einsum("bhwnc,wkc->bnhwk", q, Rw)

    # attn: bhwk b, q_h, q_w, k_h, k_w
    rel_pos = (
        rel_h[:, :, :, :, :, None] + 
        rel_w[:, :, :, :, None, :]
    ).reshape(-1, num_heads, q_h * q_w, k_h * k_w)

    return rel_pos

# modified from detectron2.modeling.backbone.utils.add_decomposed_rel_pos
def get_decomposed_rel_pos_3d(
    q: torch.Tensor,
    q_thw: Tuple[int, int, int],
    k_thw: Tuple[int, int, int],
    rel_pos_t: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    q_t, q_h, q_w = q_thw
    k_t, k_h, k_w = k_thw
    
    Rt = get_rel_pos(q_t, k_t, rel_pos_t, pad=False) 
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    
    q = q.unflatten(1, q_thw)
    rel_t_q = torch.einsum("bthwnc,tkc->bnthwk", q, Rt)
    rel_h_q = torch.einsum("bthwnc,hkc->bnthwk", q, Rh) 
    rel_w_q = torch.einsum("bthwnc,wkc->bnthwk", q, Rw) 
    
    # Combine rel pos.
    rel_pos = (
        rel_t_q[:, :, :, :, :, :, None, None] +
        rel_h_q[:, :, :, :, :, None, :, None] +
        rel_w_q[:, :, :, :, :, None, None, :]
    ).reshape(-1, num_heads, q_t*q_h*q_w, k_t*k_h*k_w)

    return rel_pos

# modified from detectron2.modeling.backbone.ViT
class CrossAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        q_thw,
        kv_thw,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # decomposed relative positional embeddings
        self.rel_pos_t = nn.Parameter(torch.zeros(2 * max(q_thw[0], kv_thw[0]) - 1, head_dim))
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * max(q_thw[1], kv_thw[1]) - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * max(q_thw[2], kv_thw[2]) - 1, head_dim))

    def forward(self, q, k, q_thw, kv_thw):
        B, _, C = q.shape
        C //= self.num_heads
        q = self.q_proj(q).reshape(B, -1, self.num_heads, C) 
        v = self.v_proj(k).reshape(B, -1, self.num_heads, C)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, C)
        
        attn_bias = get_decomposed_rel_pos_3d(
            q,
            q_thw,
            kv_thw,
            self.rel_pos_t,
            self.rel_pos_h,
            self.rel_pos_w,
            self.num_heads,
        )
        
        x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias).flatten(2)
        x = self.proj(x)
        
        return x

# modified from detectron2.modeling.backbone.ViT
class SelfAttention2D(nn.Module):
    def __init__(self, dim, num_heads, max_hw):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * max_hw[0] - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * max_hw[1] - 1, head_dim))
    
    def forward(self, x, x_hw):
        q, k, v = self.qkv(x).reshape(*x.shape[:2], 3, self.num_heads, -1).unbind(2)
        rel_pos = get_decomposed_rel_pos_2d(q, self.rel_pos_h, self.rel_pos_w, x_hw, x_hw, self.num_heads)
        x = xops.memory_efficient_attention(q, k, v, attn_bias=rel_pos).flatten(2)
        x = self.proj(x)
        return x
    
# modified from detectron2.modeling.backbone.ViT.Block
class Cross3DBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        q_thw=None,
        kv_thw=None,
    ):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.attn = CrossAttention3D(dim, num_heads, q_thw, kv_thw)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, q, k, q_thw, kv_thw):
        shortcut = q
        q = self.norm_q(q)
        k = self.norm_k(k)
        x = self.attn(q, k, q_thw, kv_thw)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# modified from detectron2.modeling.backbone.ViT.Block
class Self2DCross3DBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        max_q_thw=None,
        max_kv_thw=None,
    ):
        super().__init__()
        self.norm_self_q = norm_layer(dim)
        self.self_attn = SelfAttention2D(dim, num_heads, max_q_thw[1:])

        self.norm_cross_q = norm_layer(dim)
        self.norm_cross_k = norm_layer(dim)
        self.cross_attn = CrossAttention3D(dim, num_heads, max_q_thw, max_kv_thw)

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, q, k, q_thw, kv_thw):
        q = q + self.drop_path(self.self_attn(self.norm_self_q(q), q_thw[1:]))
        q = q + self.drop_path(self.cross_attn(self.norm_cross_q(q), self.norm_cross_k(k), q_thw, kv_thw))
        q = q + self.drop_path(self.mlp(self.norm_mlp(q)))
        return q
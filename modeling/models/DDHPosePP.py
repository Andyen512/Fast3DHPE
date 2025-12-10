import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from remote_pdb import set_trace
# from .ddhpose_hstdenoiser import HSTDenoiser

import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis

__all__ = ["DDHPosePP"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise_dir', 'pred_noise_bone', 'pred_x_start'])

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False,  bonechain=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.bonechain = bonechain


    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)       
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_xxc(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False, bonechain=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.bonechain = bonechain


    def forward(self, x, xc=None, vis=False,
                Hstack=None, hop_logits_attn=None, rel_alpha=None):
        bs, n_tok, c_dim = x.shape

        qkv = self.qkv(x).reshape(bs, n_tok, 3, self.num_heads, c_dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # [bs,H,N,d]

        if self.comb:
            logits = (q.transpose(-2, -1) @ k) * self.scale   # [bs,H,N,N]
        else:
            logits = (q @ k.transpose(-2, -1)) * self.scale   # [bs,H,N,N]

        # ---- 多跳关系偏置（注意变量名不要叫 B）----
        if (Hstack is not None) and (hop_logits_attn is not None):
            Hstack = Hstack.to(logits.device)                      # [K,N,N]
            H = logits.shape[1]
            if hop_logits_attn.dim() == 1:
                w = torch.softmax(hop_logits_attn.to(logits.device), dim=0)[None, :].expand(H, -1)  # [H,K]
            else:
                w = torch.softmax(hop_logits_attn.to(logits.device), dim=-1)                        # [H,K]
            B_head = torch.einsum('hk,kij->hij', w, Hstack)   # [H,N,N]
            if rel_alpha is not None:
                alpha = rel_alpha.to(logits.device).view(H,1,1)   # [H,1,1]
                attn_bias = (alpha * B_head).unsqueeze(0)         # [1,H,N,N]
            else:
                attn_bias = B_head.unsqueeze(0)                   # [1,H,N,N]
            logits = logits + attn_bias
        # ---- 偏置结束 ----

        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        else:
            x = (attn @ v).transpose(1, 2).contiguous().reshape(bs, n_tok, c_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False, bonechain=None):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis, bonechain=bonechain)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis


    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x
    
class Block_xxc(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention_xxc, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False, bonechain=None):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis, bonechain=bonechain)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis


    def forward(self, x, xc=None, vis=False,
                Hstack=None, hop_logits_attn=None, rel_alpha=None):
        # 注意：self.attn 的 forward 里已经按我们之前说的支持这三个参数，
        # 并在 softmax 之前把偏置加到 logits 上

        if xc is None:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x),
                    vis=vis,
                    Hstack=Hstack,
                    hop_logits_attn=hop_logits_attn,
                    rel_alpha=rel_alpha,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x),
                    self.norm1(xc),
                    vis=vis,
                    Hstack=Hstack,                # 直接透传
                    hop_logits_attn=hop_logits_attn,
                    rel_alpha=rel_alpha,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t'); x = self.reduction(x); x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t'); x = self.improve(x);   x = rearrange(x, 'b c t -> b t c')
        return x

class SinusoidalPositionEmbeddings(nn.Module):
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

class  KHSTDenoiser(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None,  boneindex=None, rootidx=0):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        
        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )

        # ---------- 构图：无向、无自环 ----------
        def chains_to_pairs(chains):
            pairs = set()
            for chain in chains:
                for i in range(len(chain)-1):
                    a, b = chain[i], chain[i+1]
                    pairs.add((a, b)); pairs.add((b, a))   # 无向
            return sorted(list(pairs))

        self.group = nn.Parameter(torch.zeros(1, 6, embed_dim))
        if rootidx==0:
            self.lev0_list = [0]
            self.lev1_list = [1,4,7]
            self.lev2_list = [2,5,8]
            self.lev3_list = [3,6,9,11,14]
            self.lev4_list = [10,12,15]
            self.lev5_list = [13,16]
            bonechain = [[0,1,2,3],[0,4,5,6],[0,7,8,9,10],[0,7,8,11,12,13],[0,7,8,14,15,16]]
            bone_pairs = chains_to_pairs([
                [0,1,2,3],
                [0,4,5,6],
                [0,7,8,9,10],
                [0,7,8,11,12,13],
                [0,7,8,14,15,16],
            ])
        else:
            self.lev0_list = [14]
            self.lev1_list = [8,11,15]
            self.lev2_list = [1,9,12,]
            self.lev3_list = [0,2,5,10,13,16]
            self.lev4_list = [3,6]
            self.lev5_list = [4,7]           
            bonechain = [[14,8,9,10],[14,11,12,13],[14,15,1,0],[14,15,1,2,3,4],[14,15,1,5,6,7],[14,15,1,16]]
            bone_pairs = chains_to_pairs([
                [14,8,9,10],
                [14,11,12,13],
                [14,15,1,0],
                [14,15,1,2,3,4],
                [14,15,1,5,6,7],
                [14,15,1,16]
            ])

        self.boneindex = boneindex


        N = 17
        M = torch.zeros(N, N)
        for i, j in bone_pairs:
            if i != j:
                M[i, j] = 1.0

        # ---------- hop 掩码（互斥的 H1/H2/H3）+ 计算 ≥4-hop ----------
        def hop_masks_with_gt(M, K=3):
            A = (M > 0).float()       # 一跳（无自环）
            hops = []
            reach = torch.eye(A.size(0))  # 已可达（含 0-hop 自己）
            P = A.clone()                 # 当前幂（k-hop 内的可达）

            for _ in range(1, K+1):
                Rk = (P > 0).float()                             # ≤k-hop
                Hk = (Rk - (reach > 0).float()).clamp(min=0.0)   # 正好 k-hop
                Hk.fill_diagonal_(0.0)
                hops.append(Hk)
                reach = ((reach > 0) | (Rk > 0)).float()         # 累计 ≤k-hop
                P = (P @ A > 0).float()                          # 下一跳

            # 传递闭包：求“所有可达”
            R_all = reach.clone()
            P_tc = P.clone()
            for _ in range(N):  # 上界 N 即可收敛（N=17）
                R_all_next = ((R_all > 0) | (P_tc > 0)).float()
                if torch.equal(R_all_next, R_all): break
                R_all = R_all_next
                P_tc = (P_tc @ A > 0).float()

            # ≥(K+1)-hop = 所有可达 - (I + H1 + ... + HK)
            sum_low = torch.eye(N)
            for Hk in hops:
                sum_low = ((sum_low > 0) | (Hk > 0)).float()
            Hgt = (R_all - sum_low).clamp(min=0.0)
            Hgt.fill_diagonal_(0.0)

            return (*hops, Hgt)   # H1, H2, H3, Hgt

        H1, H2, H3, Hgt = hop_masks_with_gt(M, K=3)

        # ---- 关键变化：不做 row_normalize，不注册 A*，而是注册 Hstack ----
        self.register_buffer("Hstack", torch.stack([H1, H2, H3, Hgt], dim=0), persistent=False)  # [4,N,N]

        # 每个 head 的 hop 权重（4 桶），softmax 后相加为 1
        self.hop_logits_attn = nn.Parameter(torch.zeros(num_heads, 4))    # [H,4]
        # 每个 head 的整体强度 α（初始化小一点更稳）
        self.rel_alpha = nn.Parameter(0.1 * torch.ones(num_heads))         # [H]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks_0 = nn.ModuleList([            
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, bonechain=bonechain)])
        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block_xxc(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, bonechain=bonechain)
            for i in range(1,depth)])

        self.TTEblocks_0 = nn.ModuleList([            
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=False, currentdim=1, depth=depth, bonechain=bonechain)])
        self.TTEblocks = nn.ModuleList([
            Block_xxc(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth, bonechain=bonechain)
            for i in range(1,depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head_pose = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def STE_forward(self, x_2d, x_3d, t):

        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            x = rearrange(x, 'b f n c  -> (b f) n c', )
            ### now x is [batch_size, receptive frames, joint_num, 2 channels]

            x = self.Spatial_patch_to_embedding(x)

            # Hierarchical embedding.
            for lev in range(6):
                lev_list = eval('self.lev{:}_list'.format(lev))
                for idx in lev_list:
                    x[:,idx,:] += self.group[0][lev:lev+1]

            x += self.Spatial_pos_embed

            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1,f,n,1)
            time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size
            x = rearrange(x, 'b h f n c  -> (b h f) n c', )

            x = self.Spatial_patch_to_embedding(x)

            # Hierarchical encoding.
            for lev in range(6):
                lev_list = eval('self.lev{:}_list'.format(lev))
                for idx in lev_list:
                    x[:,idx,:] += self.group[0][lev:lev+1]
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)

        blk = self.STEblocks_0[0]
        x = blk(x)
        # # 给空间注意力传入多跳关系先验
        # x = blk(
        #     x,
        #     Hstack=self.Hstack,                   # [4, N, N]：H1/H2/H3/Hgt
        #     hop_logits_attn=self.hop_logits_attn, # [num_heads, 4] 或 [4]
        #     rel_alpha=self.rel_alpha,             # [num_heads]
        # )
        # x = blk(x, vis=True)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)        

        blk = self.TTEblocks_0[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape

        for i in range(0, self.block_depth-1):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            # 给空间注意力传入多跳关系先验
            x = steblock(
                x,
                Hstack=self.Hstack,                   # [4, N, N]：H1/H2/H3/Hgt
                hop_logits_attn=self.hop_logits_attn, # [num_heads, 4] 或 [4]
                rel_alpha=self.rel_alpha,             # [num_heads]
            )
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            # 时间注意力先不加先验（做空间消融更稳）
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

        return x

    # 在经过STE（把STE中包含的HRST去掉）和TTE后分出xc， 后7层都用同样的Block_xxc
    def forward(self, x_2d, x_3d_dir, x_3d_bone, t, istrain):
        self.is_train = istrain
        x_3d = torch.cat((x_3d_dir,x_3d_bone), dim=-1)
        if self.is_train:
            b, f, n, c = x_2d.shape
        else:
            b, h, f, n, c = x_3d.shape
            
        x_2d, t = x_2d.float(), t.float()

        x = self.STE_forward(x_2d, x_3d, t,)

        x = self.TTE_foward(x)

        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

        x = self.ST_foward(x)

        x_pos = self.head_pose(x)
        if self.is_train:
            x_pos = x_pos.view(b, f, n, -1)
        else:
            x_pos = x_pos.view(b, h, f, n, -1)

        return x_pos

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def getbonedirect(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[1]] - seq[:,index[0]])
    bonedirect = torch.stack(bone,1)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(2), 0.5).unsqueeze(2)
    bonedirect = bonedirect/bonesum
    bonedirect = bonedirect.view(bs,ss,-1,3)
    return bonedirect

def getbonedirect_test(seq, boneindex):
    bone = []
    for index in boneindex:
        bone.append(seq[:,:,:,index[1]] - seq[:,:,:,index[0]])
    bonedirect = torch.stack(bone,3)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(-1), 0.5).unsqueeze(-1)
    bonedirect = bonedirect/bonesum
    return bonedirect

def getbonelength(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[1]] - seq[:,index[0]])
    bone = torch.stack(bone,1)
    bone = torch.pow(torch.pow(bone,2).sum(2),0.5)
    bone = bone.view(bs,ss, bone.size(1),1)
    return bone

def getbonelength_test(seq, boneindex):
    bone = []
    for index in boneindex:
        bone.append(seq[:,:,:,index[1]] - seq[:,:,:,index[0]])
    bone = torch.stack(bone,3)
    bone = torch.pow(torch.pow(bone,2).sum(-1),0.5).unsqueeze(-1)

    return bone

class DDHPosePP(nn.Module):
    """
    Implement DDHPose
    """
    def __init__(self, num_frame=234, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True,
                 test_time_augmentation=True, timestep=1000, scale=1.0, num_proposals=1, sampling_timesteps=1, 
                 boneindextemp=None, joints_left=None, joints_right=None, rootidx=0, dataset_skeleton=None):
        super().__init__()

        self.joint_nums = num_joints
        self.frames = num_frame
        self.num_proposals = num_proposals
        self.flip = test_time_augmentation
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.is_train = is_train
        self.scale = scale
        boneindextemp = boneindextemp.split(',')
        self.boneindex = []
        for i in range(0,len(boneindextemp),2):
            self.boneindex.append([int(boneindextemp[i]), int(boneindextemp[i+1])])
        self.rootidx = rootidx

        # build diffusion
        timesteps = timestep
        self.num_timesteps = int(timesteps)
        
        #timesteps_eval = args.timestep_eval
        self.sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.timesteps = timesteps
        #self.num_timesteps_eval = int(timesteps_eval)
        


        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        #self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        drop_path_rate=0
        if is_train:
            drop_path_rate=0.1

        self.dir_bone_estimator = KHSTDenoiser(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio, 
                                              depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                              drop_path_rate=drop_path_rate, boneindex=self.boneindex, rootidx=self.rootidx)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions_dir_bone(self, x_dir, x_bone, inputs_2d, input_2d_flip, t):
        x_t_dir = torch.clamp(x_dir, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t_dir = x_t_dir / self.scale
        x_t_bone = torch.clamp(x_bone, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t_bone = x_t_bone / self.scale

        pred_pose = self.dir_bone_estimator(inputs_2d, x_t_dir, x_t_bone, t, self.is_train)

        # input 2d flip
        x_t_dir_flip = x_t_dir.clone()
        x_t_dir_flip[:, :, :, :, 0] *= -1
        x_t_dir_flip[:, :, :, self.joints_left + self.joints_right] = x_t_dir_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]
        x_t_bone_flip = x_t_bone.clone()
        x_t_bone_flip[:, :, :, self.joints_left + self.joints_right] = x_t_bone_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]

        pred_pose_flip = self.dir_bone_estimator(input_2d_flip, x_t_dir_flip, x_t_bone_flip, t, self.is_train)
        
        pred_pose_flip[:, :, :, :, 0] *= -1
        pred_pose_flip[:, :, :, self.joints_left + self.joints_right] = pred_pose_flip[:, :, :,
                                                                      self.joints_right + self.joints_left]
        pred_pos = (pred_pose + pred_pose_flip) / 2
        # pred_pos = pred_pose

        x_start_dir = getbonedirect_test(pred_pos,self.boneindex)
        x_start_dir = x_start_dir * self.scale
        x_start_dir = torch.clamp(x_start_dir, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise_dir = self.predict_noise_from_start(x_dir[:,:,:,1:,:], t, x_start_dir)

        x_start_bone = getbonelength_test(pred_pos,self.boneindex)
        x_start_bone = x_start_bone * self.scale
        x_start_bone = torch.clamp(x_start_bone, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise_bone = self.predict_noise_from_start(x_bone[:,:,:,1:,:], t, x_start_bone)

        x_start_pos = pred_pos
        x_start_pos = x_start_pos * self.scale
        x_start_pos = torch.clamp(x_start_pos, min=-1.1 * self.scale, max=1.1*self.scale)

        return ModelPrediction(pred_noise_dir, pred_noise_bone, x_start_pos)

    def ddim_sample_bone_dir(self, inputs_2d, inputs_3d, clip_denoised=True, do_postprocess=True, input_2d_flip=None):
        batch = inputs_2d.shape[0]
        jt_num = inputs_2d.shape[-2]

        dir_shape = (batch, self.num_proposals, self.frames, jt_num, 3)
        bone_shape = (batch, self.num_proposals, self.frames, jt_num, 1)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img_dir = torch.randn(dir_shape, device='cuda')
        img_bone = torch.randn(bone_shape, device='cuda')

        x_start_dir = None
        x_start_bone = None

        preds_all_pos = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long).to(inputs_2d.device)
            # self_cond = x_start if self.self_condition else None

            #print("%d/%d" % (time, total_timesteps))
            preds_pos = self.model_predictions_dir_bone(img_dir, img_bone, inputs_2d, input_2d_flip, time_cond)
            pred_noise_dir, pred_noise_bone, x_start_pos = preds_pos.pred_noise_dir, preds_pos.pred_noise_bone, preds_pos.pred_x_start

            x_start_dir = getbonedirect_test(x_start_pos,self.boneindex)
            x_start_bone = getbonelength_test(x_start_pos,self.boneindex)
            
            preds_all_pos.append(x_start_pos)

            if time_next < 0:
                img_dir = x_start_dir
                img_bone = x_start_bone
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise_dir = torch.randn_like(x_start_dir)
            noise_bone = torch.randn_like(x_start_bone)

            img_dir_t = x_start_dir * alpha_next.sqrt() + \
                    c * pred_noise_dir + \
                    sigma * noise_dir
            img_bone_t = x_start_bone * alpha_next.sqrt() + \
                    c * pred_noise_bone + \
                    sigma * noise_bone
            
            img_dir[:,:,:,:self.rootidx] = img_dir_t[:,:,:,:self.rootidx]
            img_dir[:,:,:,self.rootidx+1:] = img_dir_t[:,:,:,self.rootidx:]
            img_bone[:,:,:,:self.rootidx] = img_bone_t[:,:,:,:self.rootidx]
            img_bone[:,:,:,self.rootidx+1:] = img_bone_t[:,:,:,self.rootidx:]

        return torch.stack(preds_all_pos, dim=1)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, inputs_2d, inputs_3d, input_2d_flip=None, istrain=False, inputs_act=None):
        self.is_train = istrain
        # Prepare Proposals.
        if not self.is_train:
        # if not is_train:
            self.sampling_timesteps = default(self.sampling_timesteps, self.timesteps)
            assert self.sampling_timesteps <= self.timesteps
            self.is_ddim_sampling = self.sampling_timesteps < self.timesteps
            self.ddim_sampling_eta = 1.
            self.self_condition = False
            self.box_renewal = True
            self.use_ensemble = True
            pred_pose = self.ddim_sample_bone_dir(inputs_2d, inputs_3d, input_2d_flip=input_2d_flip)
            return pred_pose

        if self.is_train:
            x_dir, dir_noises, x_bone_length, bone_length_noises, t = self.prepare_targets(inputs_3d)
            x_dir = x_dir.float()
            x_bone_length = x_bone_length.float()

            t = t.squeeze(-1)

            pred_pose = self.dir_bone_estimator(inputs_2d, x_dir, x_bone_length, t, self.is_train)
            
            # return pred_pose
            training_feat = {
                            "mpjpe": { "pred": pred_pose, "target": inputs_3d},        
                            # "dis_mpjpe": { "pred": pred_pose, "target": inputs_3d, "boneindex": self.boneindex}, 
                            # "diff_mpjpe": { "pred": pred_pose, "target": inputs_3d},   
                        }
            return training_feat


    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(self.frames, pose_3d.shape[1], pose_3d.shape[2], device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min= -1.1 * self.scale, max= 1.1*self.scale)
        x = x / self.scale


        return x, noise, t

    def prepare_diffusion_bone_dir(self, dir, bone):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise_dir = torch.randn(self.frames, dir.shape[1], dir.shape[2], device='cuda')
        noise_bone = torch.randn(self.frames, bone.shape[1], bone.shape[2], device='cuda')

        x_start_dir = dir
        x_start_bone = bone

        x_start_dir = x_start_dir * self.scale
        x_start_bone = x_start_bone * self.scale

        # noise sample
        x_dir = self.q_sample(x_start=x_start_dir, t=t, noise=noise_dir)
        x_bone = self.q_sample(x_start=x_start_bone, t=t, noise=noise_bone)

        x_dir = torch.clamp(x_dir, min= -1.1 * self.scale, max= 1.1*self.scale)
        x_dir = x_dir / self.scale
        x_bone = torch.clamp(x_bone, min= -1.1 * self.scale, max= 1.1*self.scale)
        x_bone = x_bone / self.scale


        return x_dir, noise_dir, x_bone, noise_bone, t

    def prepare_targets(self, targets):
        device = targets.device
        diffused_dir = []
        noises_dir = []
        diffused_bone_length = []
        noises_bone_length = []
        ts = []
        
        targets_dir = torch.zeros(targets.shape[0],targets.shape[1],targets.shape[2],3).to(device)
        targets_bone_length = torch.zeros(targets.shape[0],targets.shape[1],targets.shape[2],1).to(device)
        dir = getbonedirect(targets,self.boneindex)
        bone_length = getbonelength(targets,self.boneindex)
        targets_dir[:,:,:self.rootidx] = dir[:,:,:self.rootidx]
        targets_dir[:,:,self.rootidx+1:] = dir[:,:,self.rootidx:]
        targets_bone_length[:,:,:self.rootidx] = bone_length[:,:,:self.rootidx]
        targets_bone_length[:,:,self.rootidx+1:] = bone_length[:,:,self.rootidx:]

        for i in range(0,targets.shape[0]):
            targets_per_sample_dir = targets_dir[i]
            targets_per_sample_bone_length = targets_bone_length[i]

            d_dir, d_noise_dir, d_bone_length, d_noise_bone_length, d_t = self.prepare_diffusion_bone_dir(targets_per_sample_dir, targets_per_sample_bone_length)

            diffused_dir.append(d_dir)
            noises_dir.append(d_noise_dir)

            diffused_bone_length.append(d_bone_length)
            noises_bone_length.append(d_noise_bone_length)
            ts.append(d_t)

        return torch.stack(diffused_dir), torch.stack(noises_dir),  \
            torch.stack(diffused_bone_length), torch.stack(noises_bone_length), torch.stack(ts)



import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import time


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import scipy.sparse as sp
from remote_pdb import set_trace

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)

def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx) #+ sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    
    adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx

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


class LearnableGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        # spatial/temporal local topology
        self.adj = adj
        
        # simulated spatial/temporal global topology
        self.adj2 = nn.Parameter(torch.ones_like(adj))        
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        
        adj = self.adj.to(input.device) + self.adj2.to(input.device)
  
        adj = (adj.T + adj)/2
        
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class KPA(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(KPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x



class TPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, p_dropout=None):
        super(TPA, self).__init__()

        self.gconv =  LearnableGraphConv(input_dim, output_dim, adj_temporal)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class StackedTPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, hid_dim, p_dropout):
        super(StackedTPA, self).__init__()

        self.gconv1 = TPA(adj_temporal, input_dim, hid_dim, p_dropout)
        self.gconv2 = TPA(adj_temporal, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
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
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class KPAttention(nn.Module):
    def __init__(self, adj, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.kpa = KPA(adj, 2, dim, p_dropout=None)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 17, dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, vis=False):

        x = self.kpa(x)

        # x += self.Spatial_pos_embed
        x = x + self.Spatial_pos_embed

        x = self.pos_drop(x)

        res = x.clone() 

        x = self.norm1(x)

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
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        x = res + self.drop_path(x)

        return x




class TPAttention(nn.Module):
    def __init__(self, adj_temporal, num_frame, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.tpa = StackedTPA(adj_temporal, dim, dim, dim, p_dropout=None)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, vis=False):

        x = self.tpa(x)

        # x += self.Temporal_pos_embed
        x = x + self.Temporal_pos_embed

        x = self.pos_drop(x)

        res = x.clone() 

        x = self.norm1(x)

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
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = res + self.drop_path(x)

        return x




class KPA_Block(nn.Module):

    def __init__(self, adj, dim, num_heads, mlp_ratio=4., attention=KPAttention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        # self.norm1 = norm_layer(dim)
        self.attn = attention(
            adj, dim, num_heads=num_heads, drop_path=drop_path, drop_rate=drop, norm_layer=nn.LayerNorm, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
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
        # x = x + self.drop_path(self.attn(x, vis=vis))

        x = self.attn(x, vis=vis)

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




class TPA_Block(nn.Module):

    def __init__(self, adj_temporal, num_frame, dim, num_heads, mlp_ratio=4., attention=TPAttention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        # self.norm1 = norm_layer(dim)
        self.attn = attention(
            adj_temporal, num_frame, dim, num_heads=num_heads, drop_path=drop_path, drop_rate=drop, norm_layer=nn.LayerNorm, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
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
        # x = x + self.drop_path(self.attn(x, vis=vis))
        x = self.attn(x, vis=vis)

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






class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
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


class  KTPFormer_Block(nn.Module):
    def __init__(self, adj, adj_temporal, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
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

        self.pos_drop = nn.Dropout(p=drop_rate)

        num_drop_path_rate = depth + 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_drop_path_rate)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
             Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+1], norm_layer=norm_layer)
            for i in range(depth)])



        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+1], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)



        self.kpattention = KPA_Block(
                adj, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, attention=KPAttention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        

        self.tpattention = TPA_Block(
                adj_temporal, num_frame, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, attention=TPAttention, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def KPA_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        

        x = self.kpattention(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TPA_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape


        x = self.tpattention(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            # x += self.Spatial_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = steblock(x, vis=True)
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            # x += self.Temporal_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = tteblock(x, vis=True)
            #     exit()
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        # x = rearrange(x, 'b f n cw -> (b n) f cw', n=n)
        # x = self.weighted_mean(x)
        # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # x = x.view(b, f, -1)
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # x shape:(b f n c)
        # torch.cuda.synchronize()
        # st = time.time()
        x = self.KPA_forward(x)
        # now x shape is (b n) f cw
        # et = time.time()
        # print('STE_forward  ', (et-st)*2000)

        # st = time.time()
        x = self.TPA_foward(x)
        # et = time.time()
        # print('TTE_foward  ', (et-st)*2000)

        # now x shape is (b n) f cw
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # st = time.time()
        x = self.ST_foward(x)
        # et = time.time()
        # print('ST_foward  ', (et-st)*2000)

        # st = time.time()
        x = self.head(x)
        # et = time.time()
        # print('head  ', (et-st)*2000)
        # now x shape is (b f (n * 3))

        x = x.view(b, f, n, -1)

        return x



class  KTPFormer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, 
                 joints_left=None, joints_right=None, rootidx=0, dataset_skeleton=None):
        super().__init__()  

        temporal_skeleton = list(range(0, num_frame))
        temporal_skeleton = np.array(temporal_skeleton)
        temporal_skeleton -= 1
        adj_temporal = self.adj_mx_from_skeleton_temporal(num_frame, temporal_skeleton)
        adj = adj_mx_from_skeleton(dataset_skeleton)

        self.model_pos = KTPFormer_Block(adj, adj_temporal, num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                        num_heads, mlp_ratio, qkv_bias, qk_scale,
                        drop_rate, attn_drop_rate, drop_path_rate,  norm_layer)
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.rootidx = rootidx

    def adj_mx_from_edges(self, num_pts, edges, sparse=True):
        edges = np.array(edges, dtype=np.int32)
        data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
        adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

        # build symmetric adjacency matrix
        adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
        adj_mx = normalize(adj_mx) #+ sp.eye(adj_mx.shape[0]))
        if sparse:
            adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
        else:
            adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
        
        adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

        return adj_mx

    def adj_mx_from_skeleton_temporal(self, num_frame, parents):
        num_joints = num_frame
        edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
        return self.adj_mx_from_edges(num_joints, edges, sparse=False)


    def forward(self, inputs_2d, inputs_3d, input_2d_flip=None, istrain=False):
        predicted_3d_pos = self.model_pos(inputs_2d)
        
        if input_2d_flip is not None:
            predicted_3d_pos_flip = self.model_pos(input_2d_flip)
            predicted_3d_pos_flip[..., 0] *= -1
            predicted_3d_pos_flip[:, :, self.joints_left + self.joints_right] = predicted_3d_pos_flip[:, :, self.joints_right + self.joints_left]
            # for i in range(predicted_3d_pos.shape[0]):
            #     predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2
            predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2
        
        if istrain:
            # return predicted_3d_pos
            training_feat = {
                            "mpjpe": { "pred": predicted_3d_pos, "target": inputs_3d },        # 键名要等于 cfg.LOSS[*].log_prefix
                        }
            return training_feat
        else:
            return predicted_3d_pos





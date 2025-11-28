import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import torch_dct as dct
from remote_pdb import set_trace

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class  STCFormer_backbone(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        super().__init__()  
    def __init__(self, num_block, d_time, d_joint, d_coor ):
        super(STCFormer, self).__init__()

        self.num_block = num_block
        self.d_time = d_time
        self.d_joint = d_joint
        self.d_coor = d_coor

        self.stc_block = []
        for l in range(self.num_block):
            self.stc_block.append(STC_BLOCK(self.d_time, self.d_joint, self.d_coor))
        self.stc_block = nn.ModuleList(self.stc_block)

    def forward(self, input):
        # blocks layers
        for i in range(self.num_block):
            input = self.stc_block[i](input)
        # exit()
        return input

class  STCFormer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  number_of_kept_frames=3, number_of_kept_coeffs=3,
                 norm_layer=None, joints_left=None, joints_right=None, rootidx=0, dataset_skeleton=None):
        super().__init__()  

        self.model_pos = STCFormer_backbone(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                        num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.rootidx = rootidx
        
        
        
    def forward(self, inputs_2d, inputs_3d, input_2d_flip=None, istrain=False, inputs_act=None):
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
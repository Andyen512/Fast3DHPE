import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import torch_dct as dct
from remote_pdb import set_trace




class STC_ATTENTION(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, head=8):
        super().__init__()
        # print(d_time, d_joint, d_coor,head)
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)

        self.scale = (d_coor // 2) ** -0.5
        self.proj = nn.Linear(d_coor, d_coor)
        self.d_time = d_time
        self.d_joint = d_joint
        self.head = head

        # sep1
        # print(d_coor)
        self.emb = nn.Embedding(5, d_coor//head//2)
        self.register_buffer(
            "part",
            torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4], dtype=torch.long)
        )

        # sep2
        self.sep2_t = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
        self.sep2_s = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)

        self.drop = DropPath(0.5)

    def forward(self, input):
        b, t, s, c = input.shape

        h = input
        x = self.layer_norm(input)

        qkv = self.qkv(x)  # b, t, s, c-> b, t, s, 3*c
        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # 3,b,t,s,c

        # space group and time group
        qkv_s, qkv_t = qkv.chunk(2, 4)  # [3,b,t,s,c//2],  [3,b,t,s,c//2]

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]  # b,t,s,c//2
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # b,t,s,c//2

        # reshape for mat
        q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head)  # b,t,s,c//2-> b*h*t,s,c//2//h
        k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head)  # b,t,s,c//2-> b*h*t,c//2//h,s

        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)  # b,t,s,c//2 -> b*h*s,t,c//2//h
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)  # b,t,s,c//2->  b*h*s,c//2//h,t

        att_s = (q_s @ k_s) * self.scale  # b*h*t,s,s
        att_t = (q_t @ k_t) * self.scale  # b*h*s,t,t

        att_s = att_s.softmax(-1)  # b*h*t,s,s
        att_t = att_t.softmax(-1)  # b*h*s,t,t

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')

        # sep2 
        sep2_s = self.sep2_s(v_s)  # b,c//2,t,s
        sep2_t = self.sep2_t(v_t)  # b,c//2,t,s
        sep2_s = rearrange(sep2_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        sep2_t = rearrange(sep2_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        # sep1
        # v_s = rearrange(v_s, 'b c t s -> (b t ) s c')
        # v_t = rearrange(v_t, 'b c t s -> (b s ) t c')
        # print(lep_s.shape)
        sep_s = self.emb(self.part).unsqueeze(0)  # 1,s,c//2//h
        sep_t = self.emb(self.part).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1,1,1,s,c//2//h

        # MSA
        v_s = rearrange(v_s, 'b (h c) t s   -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        x_s = att_s @ v_s + sep2_s + 0.0001 * self.drop(sep_s)  # b*h*t,s,c//2//h
        x_t = att_t @ v_t + sep2_t  # b*h,t,c//h                # b*h*s,t,c//2//h

        x_s = rearrange(x_s, '(b h t) s c -> b h t s c ', h=self.head, t=t)  # b*h*t,s,c//h//2 -> b,h,t,s,c//h//2 
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2 

        x_t = x_t + 1e-9 * self.drop(sep_t)

        x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
        x = rearrange(x, 'b h t s c -> b  t s (h c) ')  # b,t,s,c

        # projection and skip-connection
        x = self.proj(x)
        x = x + h
        return x


class STC_BLOCK(nn.Module):
    def __init__(self, d_time, d_joint, d_coor):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)

        self.mlp = Mlp(d_coor, d_coor * 4, d_coor)

        self.stc_att = STC_ATTENTION(d_time, d_joint, d_coor)
        self.drop = DropPath(0.0)

    def forward(self, input):
        b, t, s, c = input.shape
        x = self.stc_att(input)
        x = x + self.drop(self.mlp(self.layer_norm(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class  STCFormer_backbone(nn.Module):
    def __init__(self, num_block, d_time, d_joint, d_coor):
        super().__init__() 

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

class Model(nn.Module):
    def __init__(self, num_frame=243, num_block=4, d_time=9, num_joints=17, embed_dim_ratio=2,
                 norm_layer=None, joints_left=None, joints_right=None, rootidx=0, dataset_skeleton=None):
        super().__init__()  

        layers, d_hid, frames = num_block, embed_dim_ratio, num_frame
        num_joints_in, num_joints_out = num_joints, num_joints

        self.pose_emb = nn.Linear(2, d_hid, bias=False)
        self.gelu = nn.GELU()
        # self.stcformer = STCFormer(layers, frames, num_joints_in, d_hid)
        self.stcformer = STCFormer_backbone(layers, frames, num_joints_in, embed_dim_ratio)
        self.regress_head = nn.Linear(d_hid, 3, bias=False)

    def forward(self, x):
        # b, t, s, c = x.shape  #batch,frame,joint,coordinate
        # dimension tranfer
        x = self.pose_emb(x)
        x = self.gelu(x)
        # spatio-temporal correlation
        x = self.stcformer(x)
        # regression head
        x = self.regress_head(x)

        return x

class  STCFormer(nn.Module):
    def __init__(self, num_frame=243, num_block=4, d_time=9, num_joints=17, embed_dim_ratio=2,
                 norm_layer=None, joints_left=None, joints_right=None, rootidx=0, dataset_skeleton=None):
        super().__init__()  

        self.model_pos = Model(num_frame, num_block, d_time, num_joints, embed_dim_ratio)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import clip
from collections import namedtuple
import math
from remote_pdb import set_trace

__all__ = ["FinePOSE"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def encode_text1(text):
    with torch.no_grad():
        text = clip.tokenize(text, truncate=True).cuda()
        return text
    
pre_text_information = [
    "A person",
    "speed",
    "head",
    "body",
    "arm",
    "leg",
]

pre_text_tensor = []
for i in pre_text_information:
    tmp_text = encode_text1(i)
    pre_text_tensor.append(tmp_text)

pre_text_tensor = torch.cat(pre_text_tensor, dim=0)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

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
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        B, T, D = h.shape
        emb = emb.view(B, T, D)
        # B, 1, 2D
        emb_out = self.emb_layers(emb)[:,0:1,:]
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        query = self.query(self.norm(x)).unsqueeze(2)
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        key = key.repeat(int(B/key.shape[0]), 1, 1, 1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)

        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).unsqueeze(1)
        value = value.repeat(int(B/value.shape[0]), 1, 1, 1)
        value = value.view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class  FinePOSE_MixSTE(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True):
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
        self.is_train=is_train

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


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        self.temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)
        self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=drop_rate,
            activation="gelu")
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=4)
        self.text_ln = nn.LayerNorm(512)
        self.text_proj = nn.Sequential(
            nn.Linear(512, 512)
        )

        self.clip_text, _ = clip.load('ViT-B/32', "cpu")
        set_requires_grad(self.clip_text, False)

        self.remain_len = 4


        ctx_vectors_subject = torch.empty((7-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_subject, std=0.02)
        self.ctx_subject = nn.Parameter(ctx_vectors_subject)

        ctx_vectors_verb = torch.empty((12-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_verb, std=0.02)
        self.ctx_verb = nn.Parameter(ctx_vectors_verb)

        ctx_vectors_speed = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_speed, std=0.02)
        self.ctx_speed = nn.Parameter(ctx_vectors_speed)

        ctx_vectors_head = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_head, std=0.02)
        self.ctx_head = nn.Parameter(ctx_vectors_head)
        
        ctx_vectors_body = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_body, std=0.02)
        self.ctx_body = nn.Parameter(ctx_vectors_body)
        
        ctx_vectors_arm = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_arm, std=0.02)
        self.ctx_arm = nn.Parameter(ctx_vectors_arm)

        ctx_vectors_leg = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_leg, std=0.02)
        self.ctx_leg = nn.Parameter(ctx_vectors_leg)

    def STE_forward(self, x_2d, x_3d, t, xf_proj):

        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, f, n, c = x.shape
            x = rearrange(x, 'b f n c  -> (b f) n c', )
            x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, :]
            xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, xf_proj.shape[1])
            time_embed = time_embed + xf_proj
            time_embed = time_embed.repeat(1, f, n, 1)
            time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape
            x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, None, :]
            xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, 1, xf_proj.shape[1])
            time_embed = time_embed + xf_proj
            time_embed = time_embed.repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x, time_embed

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x


    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x

    def encode_text(self, text, pre_text_tensor):
        with torch.no_grad():
            x = self.clip_text.token_embedding(text).type(self.clip_text.dtype)
            pre_text_tensor = self.clip_text.token_embedding(pre_text_tensor).type(self.clip_text.dtype)

        learnable_prompt_subject = self.ctx_subject
        learnable_prompt_subject = learnable_prompt_subject.view(1, self.ctx_subject.shape[0], self.ctx_subject.shape[1])
        learnable_prompt_subject = learnable_prompt_subject.repeat(x.shape[0], 1, 1)
        learnable_prompt_subject = torch.cat((learnable_prompt_subject, pre_text_tensor[:, 0, :self.remain_len, :]), dim=1)

        learnable_prompt_verb = self.ctx_verb
        learnable_prompt_verb = learnable_prompt_verb.view(1, self.ctx_verb.shape[0], self.ctx_verb.shape[1])
        learnable_prompt_verb = learnable_prompt_verb.repeat(x.shape[0], 1, 1)
        learnable_prompt_verb = torch.cat((learnable_prompt_verb, x[:, :self.remain_len, :]), dim=1)

        learnable_prompt_speed = self.ctx_speed
        learnable_prompt_speed = learnable_prompt_speed.view(1, self.ctx_speed.shape[0], self.ctx_speed.shape[1])
        learnable_prompt_speed = learnable_prompt_speed.repeat(x.shape[0], 1, 1)
        learnable_prompt_speed = torch.cat((learnable_prompt_speed, pre_text_tensor[:, 1, :self.remain_len, :]), dim=1)

        learnable_prompt_head = self.ctx_head
        learnable_prompt_head = learnable_prompt_head.view(1, self.ctx_head.shape[0], self.ctx_head.shape[1])
        learnable_prompt_head = learnable_prompt_head.repeat(x.shape[0], 1, 1)
        learnable_prompt_head = torch.cat((learnable_prompt_head, pre_text_tensor[:, 2, :self.remain_len, :]), dim=1)

        learnable_prompt_body = self.ctx_body
        learnable_prompt_body = learnable_prompt_body.view(1, self.ctx_body.shape[0], self.ctx_body.shape[1])
        learnable_prompt_body = learnable_prompt_body.repeat(x.shape[0], 1, 1)
        learnable_prompt_body = torch.cat((learnable_prompt_body, pre_text_tensor[:, 3, :self.remain_len, :]), dim=1)

        learnable_prompt_arm = self.ctx_arm
        learnable_prompt_arm = learnable_prompt_arm.view(1, self.ctx_arm.shape[0], self.ctx_arm.shape[1])
        learnable_prompt_arm = learnable_prompt_arm.repeat(x.shape[0], 1, 1)
        learnable_prompt_arm = torch.cat((learnable_prompt_arm, pre_text_tensor[:, 4, :self.remain_len, :]), dim=1)

        learnable_prompt_leg = self.ctx_leg
        learnable_prompt_leg = learnable_prompt_leg.view(1, self.ctx_leg.shape[0], self.ctx_leg.shape[1])
        learnable_prompt_leg = learnable_prompt_leg.repeat(x.shape[0], 1, 1)
        learnable_prompt_leg = torch.cat((learnable_prompt_leg, pre_text_tensor[:, 5, :self.remain_len, :]), dim=1)

        x = torch.cat((learnable_prompt_subject, learnable_prompt_verb, learnable_prompt_speed, learnable_prompt_head, learnable_prompt_body, learnable_prompt_arm, learnable_prompt_leg), dim=1)

        with torch.no_grad():
            x = x + self.clip_text.positional_embedding.type(self.clip_text.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_text.transformer(x)
            x = self.clip_text.ln_final(x).type(self.clip_text.dtype)

        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def forward(self, x_2d, x_3d, t, text, pre_text_tensor):
        if self.is_train:
            b, f, n, c = x_2d.shape
        else:
            b, h, f, n, c = x_3d.shape

        xf_proj, xf_out = self.encode_text(text, pre_text_tensor)

        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj)

        x = self.temporal_cross_attn(x, xf_out, time_embed)

        x = self.TTE_foward(x)


        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)

        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x

class  FinePOSE(nn.Module):
    """
    Implement FinePOSE
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
        self.rootidx = rootidx

        # build diffusion
        timesteps = timestep
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.num_timesteps = int(timesteps)


        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = scale
        self.box_renewal = True
        self.use_ensemble = True

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
        drop_path_rate=0
        if is_train:
            drop_path_rate=0.1

        self.pose_estimator = FinePOSE_MixSTE(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio, 
                                              depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                              drop_path_rate=drop_path_rate, is_train=is_train)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, inputs_2d, t, input_text, pre_text_tensor):
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t = x_t / self.scale
        pred_pose = self.pose_estimator(inputs_2d, x_t, t, input_text, pre_text_tensor)

        x_start = pred_pose
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def model_predictions_fliping(self, x, inputs_2d, inputs_2d_flip, t, input_text, pre_text_tensor):
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t = x_t / self.scale
        x_t_flip = x_t.clone()
        x_t_flip[:, :, :, :, 0] *= -1
        x_t_flip[:, :, :, self.joints_left + self.joints_right] = x_t_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]

        pred_pose = self.pose_estimator(inputs_2d, x_t, t, input_text, pre_text_tensor)
        pred_pose_flip = self.pose_estimator(inputs_2d_flip, x_t_flip, t, input_text, pre_text_tensor)

        pred_pose_flip[:, :, :, :, 0] *= -1
        pred_pose_flip[:, :, :, self.joints_left + self.joints_right] = pred_pose_flip[:, :, :,
                                                                      self.joints_right + self.joints_left]
        pred_pose = (pred_pose + pred_pose_flip) / 2

        x_start = pred_pose
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        pred_noise = pred_noise.float()

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, inputs_2d, inputs_3d, input_text, pre_text_tensor, clip_denoised=True, do_postprocess=True):
        batch = inputs_2d.shape[0]
        shape = (batch, self.num_proposals, self.frames, 17, 3)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # original random noise
        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        preds_all=[]
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)


            preds = self.model_predictions(img, inputs_2d, time_cond, input_text, pre_text_tensor)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return preds_all

    @torch.no_grad()
    def ddim_sample_flip(self, inputs_2d, inputs_3d, input_text, pre_text_tensor, clip_denoised=True, do_postprocess=True, input_2d_flip=None):
        batch = inputs_2d.shape[0]
        shape = (batch, self.num_proposals, self.frames, 17, 3)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device='cuda')

        x_start = None
        preds_all = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long).cuda()


            preds = self.model_predictions_fliping(img, inputs_2d, input_2d_flip, time_cond, input_text, pre_text_tensor)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return torch.stack(preds_all, dim=1)
    
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(self, inputs_2d, inputs_3d, input_2d_flip=None, istrain=False, inputs_act=None):
        
        cleaned_act = [s.split(" ")[0] for s in inputs_act]
        input_text_list = []
        for act in cleaned_act:
            tmp_text = encode_text1(act)   # 假设返回的是 [1, D] 之类的 Tensor
            input_text_list.append(tmp_text)

        input_text = torch.cat(input_text_list, dim=0).to(inputs_2d.device)  # [B, D]
        pre_text_tensor_cuda = pre_text_tensor.to(inputs_2d.device)
        pre_text_tensor_train = pre_text_tensor_cuda.unsqueeze(dim=0)
        pre_text_tensor_train = pre_text_tensor_train.repeat(input_text.shape[0], 1, 1).to(inputs_2d.device)

        self.is_train = istrain
        # Prepare Proposals.
        if not self.is_train:
            if self.flip:
                results = self.ddim_sample_flip(inputs_2d, inputs_3d, input_text, pre_text_tensor_train, input_2d_flip=input_2d_flip)
            else:
                results = self.ddim_sample(inputs_2d, inputs_3d, input_text, pre_text_tensor_train)
            return results

        if self.is_train:
            x_poses, noises, t = self.prepare_targets(inputs_3d)
            x_poses = x_poses.float()
            t = t.squeeze(-1)

            # pred_pose = self.pose_estimator(inputs_2d, x_poses, t, self.is_train)
            pred_pose = self.pose_estimator(inputs_2d, x_poses, t, input_text, pre_text_tensor_train)
            # return pred_pose
            training_feat = {
                            "mpjpe": { "pred": pred_pose, "target": inputs_3d},        # 键名要等于 cfg.LOSS[*].log_prefix
                        }
            return training_feat
        

    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(self.frames, 17, 3, device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min= -1.1 * self.scale, max= 1.1*self.scale)
        x = x / self.scale


        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)
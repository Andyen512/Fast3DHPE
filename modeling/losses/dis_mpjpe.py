import torch
import torch.nn as nn
from einops import rearrange
from remote_pdb import set_trace
from .loss_utils import *

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

class Dis_MPJPELoss(nn.Module):
    """
    输入:
      pred: [B, T, J, 3]
      gt:   [B, T, J, 3]
      unit: "m" or "mm"（仅统计用；真实单位由数据决定）
    返回:
      loss: 标量张量
      info: dict（供日志）
    """
    def __init__(self, unit: str = "mm"):
        super().__init__()
        self.unit = unit
        self.loss_term_weight = 1.0



    def forward(self, pred, gt, boneindex):
        # [B,T,J,3]
        loss_3d_pos = mpjpe(pred, gt)

        gt_len = getbonelength(gt, boneindex).mean(1)
        pd_len = getbonelength(pred, boneindex).mean(1)
        loss_length = torch.pow(gt_len - pd_len, 2).mean()

        gt_dir = getbonedirect(gt, boneindex)  # [N, B, 3]
        pd_dir = getbonedirect(pred, boneindex)
        loss_dir = torch.pow(gt_dir - pd_dir, 2).sum(3).mean()

        loss_total = loss_3d_pos + loss_length + loss_dir
        
        return loss_total, {"loss_total": loss_total ,
                       "loss_3d_pos": loss_3d_pos}


import torch
import numpy as np
import torch.nn as nn

def mpjpe(predicted, target, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if not return_joints_err:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        # errors: [B, T, N]
        from einops import rearrange
        errors = rearrange(errors, 'B T N -> N (B T)')
        errors = torch.mean(errors, dim=-1).cpu().numpy().reshape(-1) * 1000
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)), errors

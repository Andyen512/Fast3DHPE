import torch
import torch.nn as nn
from einops import rearrange

class MPJPELoss(nn.Module):
    """
    支持两种输入：
    Case A (标注级 MPJPE)：
        pred: [B, T, J, 3]
        gt:   [B, T, J, 3]
        -> 返回: (loss_scalar, info)

    Case B (多采样/多头对齐误差)：
        pred: [B, T, H, F, N, 3]   # 比如 T 个采样步、H 个head、F帧、N关节
        gt:   [B, F, N, 3]         # 目标（会在 T、H 维度上广播）
        -> 返回: (loss_th, info)，其中 loss_th 形状 [T, H]
    """
    def __init__(self, unit: str = "mm", scale: float = 1.0):
        """
        unit: 仅用于日志标识；不会强制缩放（避免与你的数据单位冲突）
        scale: 若你需要强制从 m -> mm，可设为 1000.0；默认 1.0 不缩放
        """
        super().__init__()
        self.unit = unit
        self.scale = float(scale)

    def forward(self, pred, target):
        if pred.ndim == 4:
            # ---- Case A: pred, gt ∈ [B, T, J, 3] ----
            if target.ndim != 4:
                raise ValueError(f"Case A 期望 gt.dim==4，但得到 {target.ndim}")
            diff = (pred - target) * self.scale
            mpjpe = diff.norm(dim=-1).mean()  # 标量
            info = {
                "loss_total": mpjpe,
                "loss_3d_pos": mpjpe,
            }
            return mpjpe, info

        elif pred.ndim == 6:
            t = pred.shape[1]
            h = pred.shape[2]
            # print(predicted.shape)
            # print(target.shape)
            target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
            errors = torch.norm(pred - target, dim=len(target.shape)-1)
            
            #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
            errors = rearrange(errors, 'b t h f n  -> t h b f n', ).reshape(t, h, -1)
            errors = torch.mean(errors, dim=-1, keepdim=False)

            info = {
                "loss_total": errors, 
            }
            return errors, info

        else:
            raise ValueError(
                f"不支持的 pred 维度：{pred.shape}。"
                "期望 [B,T,J,3] 或 [B,T,H,F,N,3]。"
            )
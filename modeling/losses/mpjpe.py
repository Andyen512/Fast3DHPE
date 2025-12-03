import torch
import torch.nn as nn
from einops import rearrange

class MPJPELoss(nn.Module):
    """
    支持两种输入：
    Case A (标注级 MPJPE):
        pred: [B, T, J, 3]
        gt:   [B, T, J, 3]
        -> 返回: (loss_scalar, info)

    Case B (多采样/多头对齐):
        pred: [B, T, H, F, N, 3]
        gt:   [B, F, N, 3]
        -> 返回: (loss_th, info)，其中 loss_th 形状 [T, H]
    """
    def __init__(
        self,
        unit: str = "mm",
        scale: float = 1.0,
        use_weight: bool = False,
        joint_weight: torch.Tensor = None,
    ):
        """
        unit: for logging only, no scaling applied
        scale: 1 -> no scaling; 1000 converts m->mm
        use_weight: whether to apply joint weights
        joint_weight: tensor of shape [J], e.g. w_mpjpe
        """
        super().__init__()
        self.unit = unit
        self.scale = float(scale)
        self.use_weight = use_weight

        if joint_weight is not None:
            # self.register_buffer("joint_weight", joint_weight.float())
            self.joint_weight = torch.tensor(joint_weight)
        else:
            self.joint_weight = None

    def forward(self, pred, target):
        # -------------------------------------------------------------
        # Case A: pred ∈ [B,T,J,3], target ∈ [B,T,J,3]
        # -------------------------------------------------------------
        if pred.ndim == 4:
            if target.ndim != 4:
                raise ValueError(f"Case A expects target dim=4, got {target.ndim}")
            diff = (pred - target) * self.scale     # [B,T,J,3]
            dist = diff.norm(dim=-1)                # [B,T,J]

            loss_3d_pos = dist.mean()  # [B,T]
            # ---- apply joint weight if enabled ----
            if self.use_weight:
                if self.joint_weight is None:
                    raise ValueError("joint_weight is None but use_weight=True")
                # broadcast to [B,T,J]
                weight = self.joint_weight.view(1,1,-1).to(pred.device)
                dist = dist * weight

            mpjpe = dist.mean()   # scalar

            info = {
                "loss_total": mpjpe,
                "loss_3d_pos": loss_3d_pos,
            }
            return mpjpe, info

        # -------------------------------------------------------------
        # Case B: pred ∈ [B,T,H,F,N,3], target ∈ [B,F,N,3]
        # -------------------------------------------------------------
        elif pred.ndim == 6:
            B, T, H, F, N, _ = pred.shape

            # expand target → [B,T,H,F,N,3]
            target = target.unsqueeze(1).unsqueeze(1)
            target = target.repeat(1, T, H, 1, 1, 1)

            diff = (pred - target) * self.scale      # [B,T,H,F,N,3]
            dist = diff.norm(dim=-1)                 # [B,T,H,F,N]

            # ---- apply joint weight if enabled ----
            if self.use_weight:
                if self.joint_weight is None:
                    raise ValueError("joint_weight is None but use_weight=True")
                # weight: [N] → reshape to [1,1,1,1,N]
                weight = self.joint_weight.view(1,1,1,1,N)
                dist = dist * weight                 # weighted per joint

            # rearrange → [T,H,B,F,N] then compute mean over last dim
            dist = rearrange(dist, 'b t h f n -> t h b f n')
            dist = dist.reshape(T, H, -1)            # [T,H, B*F*N_weighted]
            loss_th = dist.mean(dim=-1)              # [T, H]

            info = {"loss_total": loss_th}
            return loss_th, info

        else:
            raise ValueError(
                f"Unsupported pred shape {pred.shape}. "
                "Expected [B,T,J,3] or [B,T,H,F,N,3]."
            )
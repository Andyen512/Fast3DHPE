import torch
import torch.nn as nn

def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for 3D pose sequences.

    Inputs:
        pred: [B, T, J, 3]  - predicted 3D poses
        target: [B, T, J, 3]  - ground-truth 3D poses
    """

    def __init__(self, dataset: str = "h36m", axis: int = 1):
        super().__init__()
        self.axis = axis

        if dataset.lower() == "h36m":
            w_mpjpe = torch.tensor([
                1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1,
                1.5, 1.5, 4, 4, 1.5, 4, 4
            ], dtype=torch.float32)
        elif dataset.lower() == "humaneva15":
            w_mpjpe = torch.tensor([
                1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1,
                1.5, 1.5, 4, 4, 1.5, 4, 4
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dataset '{dataset}' for TemporalConsistencyLoss.")

        self.register_buffer("w_mpjpe", w_mpjpe)
        self.loss_term_weight = 1.0

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, T, J, 3]
            target: [B, T, J, 3]
        """
        # 1) Temporal difference between consecutive frames
        dif_seq = pred[:, 1:, :, :] - pred[:, :-1, :, :]

        # 2) Build joint-wise weights with broadcasting
        weights_joints = torch.ones_like(dif_seq)
        # ensure weights are on the same device as pred
        weights_mul = self.w_mpjpe.to(pred.device)

        assert weights_mul.shape[0] == weights_joints.shape[-2], \
            f"Joint weight length {weights_mul.shape[0]} != number of joints {weights_joints.shape[-2]}"

        weights_joints = (weights_joints.permute(0, 1, 3, 2) * weights_mul).permute(0, 1, 3, 2)

        # 3) Weighted temporal smoothness term
        temporal_term = torch.mean(weights_joints * dif_seq.pow(2))

        # 4) Mean velocity error term
        vel_term = mean_velocity_error_train(pred, target, axis=self.axis)

        # 5) Final temporal consistency loss
        loss_diff = 0.5 * temporal_term + 2.0 * vel_term

        info = {
            "loss_tc": loss_diff,
            "loss_tc_temporal": temporal_term.detach(),
            "loss_tc_velocity": vel_term.detach(),
        }
        return loss_diff, info
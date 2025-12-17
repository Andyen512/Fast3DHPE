import inspect
import torch
import torch.nn as nn
from . import losses

def is_dict(x): return isinstance(x, dict)
def is_tensor(x): return torch.is_tensor(x)

def get_attr_from(modules, name: str):
    for m in modules:
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(f"{name} not found in {modules}")

def get_valid_args(Cls, cfg: dict, drop_keys=None):
    sig = inspect.signature(Cls.__init__)
    keys = set(sig.parameters.keys()) - { "self" }
    if drop_keys:
        keys -= set(drop_keys)
    return { k: cfg[k] for k in cfg if k in keys }

def get_ddp_module(m):
    # Return the raw module; wrap losses with DDP here if needed
    return m

class Odict(dict):
    """Placeholder for an ordered dict; regular dict keeps compatibility."""
    pass


class LossAggregator(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        if is_dict(loss_cfg):
            items = {loss_cfg["log_prefix"]: self._build_loss_(loss_cfg)}
        else:
            items = {cfg["log_prefix"]: self._build_loss_(cfg) for cfg in loss_cfg}
        self.losses = nn.ModuleDict(items)

    def _build_loss_(self, cfg):
        Loss = get_attr_from([losses], cfg["type"])
        valid = get_valid_args(Loss, cfg, drop_keys=["type", "gather_and_scale", "log_prefix", "loss_term_weight"])
        loss = Loss(**valid)

        loss.loss_term_weight = float(cfg.get("loss_term_weight", 1.0))

        return loss

    def forward(self, training_feats: dict):
        """
        training_feats: model output dict; keys must match loss_cfg log_prefix
          Example:
            {
              "mpjpe": { "pred": ..., "gt": ... },
              "bone":  { "pred": ..., "gt": ..., "bone_index": ... },
            }
        """
        loss_sum = training_feats[":_sum_"].mean()*0 if (":_sum_" in training_feats) else 0.0  # Dummy tensor to keep device alignment
        loss_info = Odict()

        for k, v in training_feats.items():
            if k in self.losses:
                # v must be a dict; pass as **kwargs to loss.forward
                if not is_dict(v):
                    raise ValueError(f"training_feats['{k}'] must be a dict of arguments for its loss.")
                loss_func = self.losses[k]
                loss, info = loss_func(**v)        # Each loss returns (scalar_loss, info_dict)
                # Scalar term
                loss = loss.mean() * loss_func.loss_term_weight
                for name, value in (info or {}).items():
                    # loss_info[f"scalar/{k}/{name}"] = value
                    loss_info[f"{name}"] = value
                loss_sum = loss_sum + loss
            else:
                # Allow feeding tensors directly into the sum
                if is_tensor(v):
                    _ = v.mean()
                    loss_info[f"scalar/{k}"] = _
                    loss_sum = loss_sum + _
                elif is_dict(v):
                    raise ValueError(f"Key '{k}' not in configured losses; add log_prefix='{k}' to cfg['LOSS'] or rename.")
                else:
                    raise ValueError(f"Unsupported training_feats['{k}'] type: {type(v)}")

        return loss_sum, loss_info
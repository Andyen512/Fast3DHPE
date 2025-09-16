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
    # 在聚合器里直接返回原模块；如果你想把损失也 DDP 包起来，改这里
    return m

class Odict(dict):
    """有序字典占位（这边直接用 dict 即可，保留名字兼容）"""
    pass


class LossAggregator(nn.Module):
    """
    OpenGait 风格：把多个 loss 统一配置、统一前向，返回 (loss_sum, loss_info)
    loss_cfg 支持 dict（单个）或 list[dict]（多个）
    """
    def __init__(self, loss_cfg) -> None:
        super().__init__()
        if is_dict(loss_cfg):
            items = { loss_cfg["log_prefix"]: self._build_loss_(loss_cfg) }
        else:
            items = { cfg["log_prefix"]: self._build_loss_(cfg) for cfg in loss_cfg }
        self.losses = nn.ModuleDict(items)

    def _build_loss_(self, cfg):
        Loss = get_attr_from([losses], cfg["type"])           # 例如 "MPJPELoss"
        valid = get_valid_args(Loss, cfg, drop_keys=["type", "gather_and_scale", "log_prefix", "loss_term_weight"])
        loss = Loss(**valid)
        # 设置权重；OpenGait 用 loss.loss_term_weight
        loss.loss_term_weight = float(cfg.get("loss_term_weight", 1.0))
        return get_ddp_module(loss.cuda())

    def forward(self, training_feats: dict):
        """
        training_feats: 模型输出的 dict；键要与 loss_cfg 的 log_prefix 对齐
          例：
            {
              "mpjpe": { "pred": ..., "gt": ... },
              "bone":  { "pred": ..., "gt": ..., "bone_index": ... },
            }
        """
        loss_sum = training_feats[":_sum_"].mean()*0 if (":_sum_" in training_feats) else 0.0  # 哑元，兼容标量设备
        loss_info = Odict()

        for k, v in training_feats.items():
            if k in self.losses:
                # v 必须是 dict，按 **kwargs 传给 loss.forward
                if not is_dict(v):
                    raise ValueError(f"training_feats['{k}'] must be a dict of arguments for its loss.")
                loss_func = self.losses[k]
                loss, info = loss_func(**v)        # 每个 loss 返回 (scalar_loss, info_dict)
                # 标量与标量项
                loss = loss.mean() * loss_func.loss_term_weight
                for name, value in (info or {}).items():
                    # loss_info[f"scalar/{k}/{name}"] = value
                    loss_info[f"{name}"] = value
                loss_sum = loss_sum + loss
            else:
                # 兼容“直接喂 into sum 的张量”
                if is_tensor(v):
                    _ = v.mean()
                    loss_info[f"scalar/{k}"] = _
                    loss_sum = loss_sum + _
                elif is_dict(v):
                    raise ValueError(f"Key '{k}' not in configured losses; add log_prefix='{k}' to cfg['LOSS'] or rename.")
                else:
                    raise ValueError(f"Unsupported training_feats['{k}'] type: {type(v)}")

        return loss_sum, loss_info
import os, torch
from torch.cuda.amp import autocast, GradScaler
from .dist import is_main_process
from .loss_aggregator import LossAggregator
from remote_pdb import set_trace
import torch.distributed as dist

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
from time import time
from .evaluation.evaluator import *
from torch.nn.parallel import DistributedDataParallel as DDP

def save_checkpoint(model, optimizer, scheduler, cfg, out_dir,
                    epoch, metric=None, is_best=False, tag=None, logger=None):
    """
    Generic checkpoint saving function (rank0-only)

    Args:
        model: nn.Module or a DDP-wrapped model
        optimizer: torch.optim.Optimizer
        scheduler: torch.optim.lr_scheduler._LRScheduler or None
        cfg: current config dict (kept for reproducibility)
        out_dir: str, output root directory
        epoch: int, current epoch
        metric: float or None, optional validation metric (e.g., val_loss)
        is_best: bool, whether to save as best.pth
        tag: str or None, optional filename suffix (e.g., 'epoch_10')
        logger: optional logger instance
    """
    if not is_main_process():
        return

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Strip the DDP wrapper from the model
    model_to_save = model.module if hasattr(model, "module") else model
    lr = optimizer.param_groups[0]["lr"]

    state = {
        "epoch": epoch,
        "lr": lr,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "cfg": cfg,
    }
    if metric is not None:
        state["metric"] = metric

    if is_best:
        save_path = os.path.join(ckpt_dir, "best_epoch.bin")
    elif tag is not None:
        save_path = os.path.join(ckpt_dir, f"{tag}.bin")
    else:
        save_path = os.path.join(ckpt_dir, f"epoch_{epoch}.bin")

    torch.save(state, save_path)

    if logger:
        logger.info(f"Checkpoint saved: {save_path}")
    else:
        print(f"Checkpoint saved: {save_path}")

class Trainer:
    def __init__(self, cfg, logger, out_dir, device):
        self.cfg = cfg
        self.dataset_cfg = cfg.get("DATASET", {})
        self.root_idx = self.dataset_cfg['Root_idx']
        self.logger = logger
        self.out_dir = out_dir
        loss_cfg = cfg.get("LOSS", {"type": "MPJPELoss", "log_prefix": "mpjpe", "loss_term_weight": 1.0})
        self.loss_agg = LossAggregator(loss_cfg)
        loss_eval = cfg.get("LOSS_EVAL", {"type": "MPJPELoss", "log_prefix": "mpjpe", "loss_term_weight": 1.0})
        self.loss_eval = LossAggregator(loss_eval).to(device)
        self.iteration = 0
        self.optimizer = None
        self.scheduler = None
        self.lr = None
        self.wd = None
        self.lrd = None
        self.best_rec = {"metric": float("inf"), "epoch": -1}
        self.training = None

        # ----- AMP config -----
        self.use_amp = cfg.get("ENGINE", {}).get("amp", True)  # or False if you want default off
        self.scaler = GradScaler(enabled=self.use_amp)

    def _build_optim(self, model):
        opt_cfg = self.cfg.get("OPTIM", {})
        self.lr = opt_cfg.get("lr", 1e-4)
        self.wd = opt_cfg.get("weight_decay", 0.1)
        self.lrd = opt_cfg.get("lr_decay", 0.999)
        return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)

    def train(self, model, train_loader, val_loader, bundle, ckpt_path=None):
        if train_loader is None:
            if is_main_process():
                self.logger.info("train_loader is None — fill datasets/builder.py to enable training.")
            return

        if self.optimizer is None:
            self.optimizer = self._build_optim(model)
        
        epoch = 0
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
            model.module.load_state_dict(checkpoint['model'], strict=False)
            epoch = checkpoint['epoch']
            self.best_rec["metric"] = checkpoint['metric']
            self.best_rec["epoch"] = checkpoint['epoch']
            if is_main_process():
                print('This model was trained for {} epochs'.format(checkpoint['epoch']))
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        receptive_field = self.cfg["DATASET"]["receptive_field"]
        epochs = self.cfg.get("SCHED", {}).get("max_epochs", 1)
        device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
        losses_3d_train = []
        losses_3d_pos_train = []
        losses_3d_valid = []

        while epoch < epochs+1:
            start_time = time()
            model.train()
            self.training = True
            epoch_loss_3d_train = 0.0
            epoch_loss_3d_pos_train = 0.0
            N = 0
            iteration = 0
            quickdebug = self.cfg.get("DEBUG", False)
            for cameras_train, batch_3d, batch_2d, seq_name, batch_act in train_loader:
                if cameras_train is not None:
                    cameras_train = cameras_train.float()
                inputs_3d = batch_3d.float()
                inputs_2d = batch_2d.float()
                inputs_act = batch_act

                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda(non_blocking=True)
                    inputs_2d = inputs_2d.cuda(non_blocking=True)
                    if cameras_train is not None:
                        cameras_train = cameras_train.cuda(non_blocking=True)

                inputs_3d[:, :, self.root_idx] = 0  # Root alignment

                # self.optimizer.zero_grad(set_to_none=True)
                # training_feat = model(inputs_2d, inputs_3d, None, self.training, inputs_act)
                
                # loss_total, loss_info = self.loss_agg(training_feat)
                # loss_total.backward()
                # self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                # ----- forward & loss under autocast -----
                with autocast(enabled=self.use_amp):
                    training_feat = model(inputs_2d, inputs_3d, None, self.training, inputs_act)
                    loss_total, loss_info = self.loss_agg(training_feat)
 
                # ----- backward & step with / without AMP -----
                if self.use_amp:
                    self.scaler.scale(loss_total).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_total.backward()
                    self.optimizer.step()

                bs, ts = inputs_3d.shape[0], inputs_3d.shape[1]
                epoch_loss_3d_train     += (bs * ts) * loss_total.detach().float().item()
                epoch_loss_3d_pos_train += (bs * ts) * loss_info["loss_3d_pos"].detach().float().item()
                N += (bs * ts)

                iteration += 1
                if quickdebug and N == (bs * ts):
                    break

            t = torch.tensor([epoch_loss_3d_train, epoch_loss_3d_pos_train, float(N)],
                            device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_loss_3d_train, epoch_loss_3d_pos_train, N = t[0].item(), t[1].item(), int(t[2].item())

            losses_3d_train.append(epoch_loss_3d_train / N)
            losses_3d_pos_train.append(epoch_loss_3d_pos_train / N)

            if self.cfg["EVAL"]:
                self.training = False
                model.eval()
                if isinstance(model, DDP):
                    model.module.eval()   # Triggers custom eval() -> build_eval_model()
                local_sum = torch.zeros(1, device=device, dtype=torch.float32)
                local_cnt = torch.zeros(1, device=device, dtype=torch.float32)

                with torch.inference_mode():
                    for cam, batch_3d, batch_2d, _, batch_act in val_loader:
                        if cam is not None:
                            cam = cam.float()
                        inputs_3d = batch_3d.float()
                        inputs_2d = batch_2d.float()
                        inputs_act = batch_act

                        inputs_2d_flip = inputs_2d.clone()
                        inputs_2d_flip[..., 0] *= -1
                        inputs_2d_flip[..., bundle.joints_left + bundle.joints_right, :] = inputs_2d_flip[..., bundle.joints_right + bundle.joints_left, :]

                        # Shape conversion
                        inputs_3d_p = inputs_3d
                        inputs_2d, inputs_3d      = eval_data_prepare(self.cfg['DATASET']['dataset_type'], receptive_field, inputs_2d, inputs_3d_p)
                        inputs_2d_flip, _         = eval_data_prepare(self.cfg['DATASET']['dataset_type'], receptive_field, inputs_2d_flip, inputs_3d_p)

                        if torch.cuda.is_available():
                            inputs_3d = inputs_3d.cuda(non_blocking=True)
                            inputs_2d = inputs_2d.cuda(non_blocking=True)
                            inputs_2d_flip = inputs_2d_flip.cuda(non_blocking=True)
              
                        inputs_3d[..., self.root_idx, :] = 0  # Root alignment
                        # pred_3d = model(inputs_2d, inputs_3d, inputs_2d_flip, self.training, inputs_act)

                        # # ====== Loss computation (same as training) ======
                        # training_feat = {
                        #     "mpjpe": {"pred": pred_3d, "target": inputs_3d}
                        # }
                        # loss_total, loss_info = self.loss_eval(training_feat)

                        with autocast(enabled=self.use_amp):
                            pred_3d = model(inputs_2d, inputs_3d, inputs_2d_flip, self.training, inputs_act)

                        training_feat = {
                            "mpjpe": {"pred": pred_3d, "target": inputs_3d}
                        }
                        loss_total, loss_info = self.loss_eval(training_feat)

                        bs, ts = inputs_3d.shape[0], inputs_3d.shape[1]
                        local_sum += loss_total.detach() * (bs * ts)
                        local_cnt += (bs * ts)
                        
                        del pred_3d, inputs_2d, inputs_2d_flip, inputs_3d
                        torch.cuda.empty_cache()
                        
            
                # ====== Distributed all_reduce to sync every GPU ======
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)

                valid_scalar = (local_sum / local_cnt).item() if local_cnt.item() > 0 else float("inf")
                
                elapsed = (time() - start_time) / 60.0
                
                losses_3d_valid.append(valid_scalar)

                if self.cfg["EVAL"]:
                    last_valid = float(losses_3d_valid[-1])
                    # print('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 3d_pos_valid %f' % (
                    #     epoch + 1,
                    #     elapsed,
                    #     self.lr,
                    #     losses_3d_train[-1] * 1000,
                    #     losses_3d_pos_train[-1] * 1000,
                    #     last_valid * 1000
                    # ))
                    if is_main_process():
                        self.logger.info(
                            "[%d] time %.2f lr %.6f 3d_train %.3f 3d_pos_train %.3f 3d_pos_valid %.3f" % (
                                epoch,
                                elapsed,
                                self.optimizer.param_groups[0]['lr'],
                                losses_3d_train[-1] * 1000,
                                losses_3d_pos_train[-1] * 1000,
                                last_valid * 1000
                            )
                        )

                if epoch != 0 and epoch % self.cfg["ENGINE"]["save_freq"] == 0:
                    # Periodic checkpoint save
                    save_checkpoint(model, self.optimizer, self.scheduler,
                                    self.cfg, self.out_dir, epoch, metric=last_valid * 1000, 
                                    tag=f"epoch_{epoch}", logger=self.logger)

                # Save best checkpoint
                if self.cfg["EVAL"] and last_valid * 1000 < self.best_rec["metric"]:
                    self.best_rec = {"metric": last_valid * 1000, "epoch": epoch}
                    save_checkpoint(model, self.optimizer, self.scheduler,
                                    self.cfg, self.out_dir, epoch, metric=last_valid * 1000,
                                    is_best=True, logger=self.logger)
                    
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lrd
            epoch += 1


    def test(self, cfg, model, bundle_list, ckpt_path=None):
        eval_type         = cfg["DATASET"]["Test"]["Eval_type"]
        test_dataset_name = cfg["DATASET"].get("test_dataset", "")

        if self.optimizer is None:
            self.optimizer = self._build_optim(model)

        # ---------- load ckpt ----------
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
            model.module.load_state_dict(checkpoint['model'], strict=False)
            epoch = checkpoint['epoch']
            if is_main_process():
                print('This model was trained for {} epochs'.format(checkpoint['epoch']))
                print('Loading evaluate checkpoint')
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        # =====================================================
        # Special branch: 3DHP (single overall eval for JPMA/Normal)
        # =====================================================
        if test_dataset_name == '3DHP':
            if len(bundle_list) == 0:
                if is_main_process():
                    self.logger.info("No test bundle found for 3DHP.")
                return

            bundle      = bundle_list[0]   # Convention: 3DHP has one overall bundle
            test_loader = bundle.test_loader
            kps_left    = bundle.kps_left
            kps_right   = bundle.kps_right
            joints_left = bundle.joints_left
            joints_right= bundle.joints_right

            if test_loader is None:
                if is_main_process():
                    self.logger.info("test_loader is None — fill datasets/builder.py to enable testing.")
                return

            if eval_type == 'JPMA':
                # 3DHP + JPMA:
                #  - Step-wise MPJPE logs already printed in evaluate
                #  - evaluate also saves all four hypotheses as .mat
                if cfg['DATASET']['Test']['P2']:
                    _ = evaluate(
                        cfg, test_loader, model_pos=model,
                        kps_left=kps_left, kps_right=kps_right,
                        joints_left=joints_left, joints_right=joints_right,
                        action=None, logger=self.logger
                    )
                else:
                    _ = evaluate(
                        cfg, test_loader, model_pos=model,
                        kps_left=kps_left, kps_right=kps_right,
                        joints_left=joints_left, joints_right=joints_right,
                        action=None, logger=self.logger
                    )

                # evaluate already logged and saved .mat; return directly
                return

            elif eval_type == 'Normal':
                # 3DHP + Normal: single evaluation plus PCK/AUC
                e1, e2, e3, ev, pck, auc = evaluate(
                    cfg, test_loader, model_pos=model,
                    kps_left=kps_left, kps_right=kps_right,
                    joints_left=joints_left, joints_right=joints_right,
                    action=None, logger=self.logger
                )

                import torch as _torch

                def _to_float(x):
                    if x is None:
                        return None
                    return x.item() if isinstance(x, _torch.Tensor) else float(x)

                e1_f  = _to_float(e1)
                e2_f  = _to_float(e2)
                e3_f  = _to_float(e3)
                ev_f  = _to_float(ev)
                pck_f = _to_float(pck)
                auc_f = _to_float(auc)

                if is_main_process():
                    self.logger.info("========== 3DHP overall evaluation ==========")
                    self.logger.info("Protocol #1 Error (MPJPE):   %.4f mm", e1_f)
                    self.logger.info("Protocol #2 Error (P-MPJPE): %.4f mm", e2_f)
                    self.logger.info("Protocol #3 Error (N-MPJPE): %.4f mm", e3_f)
                    self.logger.info("Velocity Error (MPJVE):      %.4f mm", ev_f)
                    if pck_f is not None and auc_f is not None:
                        self.logger.info("3DHP PCK (overall):         %.2f",  pck_f)
                        self.logger.info("3DHP AUC (overall):         %.4f", auc_f)
                    self.logger.info("=============================================")
                return

        # =====================================================
        # General branch: datasets like H36M keep per-action evaluation + averaging
        # =====================================================
        if eval_type == 'JPMA':
            errors_p1        = []
            errors_p1_h      = []
            errors_p1_mean   = []
            errors_p1_select = []

            errors_p2        = []
            errors_p2_h      = []
            errors_p2_mean   = []
            errors_p2_select = []
        elif eval_type == 'Normal':
            errors_p1  = []
            errors_p2  = []
            errors_p3  = []
            errors_vel = []
            errors_pck, errors_auc = [], []

        for bundle in bundle_list:
            test_loader = bundle.test_loader
            action_key  = bundle.action_key
            action_filter = bundle.action_filter
            kps_left    = bundle.kps_left
            kps_right   = bundle.kps_right
            joints_left = bundle.joints_left
            joints_right= bundle.joints_right

            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            if test_loader is None:
                if is_main_process():
                    self.logger.info("train_loader is None — fill datasets/builder.py to enable training.")
                return

            if eval_type == 'JPMA':
                if cfg['DATASET']['Test']['P2']:
                    e1, e1_h, e1_mean, e1_select, \
                    e2, e2_h, e2_mean, e2_select = evaluate(
                        cfg, test_loader, model_pos=model,
                        kps_left=kps_left, kps_right=kps_right,
                        joints_left=joints_left, joints_right=joints_right,
                        action=action_key, logger=self.logger
                    )
                else:
                    e1, e1_h, e1_mean, e1_select = evaluate(
                        cfg, test_loader, model_pos=model,
                        kps_left=kps_left, kps_right=kps_right,
                        joints_left=joints_left, joints_right=joints_right,
                        action=action_key, logger=self.logger
                    )

                errors_p1.append(e1)
                errors_p1_h.append(e1_h)
                errors_p1_mean.append(e1_mean)
                errors_p1_select.append(e1_select)

                if cfg['DATASET']['Test']['P2']:
                    errors_p2.append(e2)
                    errors_p2_h.append(e2_h)
                    errors_p2_mean.append(e2_mean)
                    errors_p2_select.append(e2_select)

            elif eval_type == 'Normal':
                # All datasets under Normal unpack six return values;
                # For non-3DHP datasets evaluate returns pck=auc=None
                e1, e2, e3, ev = evaluate(
                    cfg, test_loader, model_pos=model,
                    kps_left=kps_left, kps_right=kps_right,
                    joints_left=joints_left, joints_right=joints_right,
                    action=action_key, logger=self.logger
                )

                errors_p1.append(torch.as_tensor(e1, dtype=torch.float32))
                errors_p2.append(torch.as_tensor(e2, dtype=torch.float32))
                errors_p3.append(torch.as_tensor(e3, dtype=torch.float32))
                errors_vel.append(torch.as_tensor(ev, dtype=torch.float32))


        # ---------- JPMA aggregation (H36M etc.) ----------
        if eval_type == 'JPMA':
            errors_p1  = torch.stack(errors_p1)
            errors_p1_actionwise = torch.mean(errors_p1, dim=0)
            errors_p1_h  = torch.stack(errors_p1_h)
            errors_p1_actionwise_h = torch.mean(errors_p1_h, dim=0)
            errors_p1_mean  = torch.stack(errors_p1_mean)
            errors_p1_actionwise_mean = torch.mean(errors_p1_mean, dim=0)
            errors_p1_select  = torch.stack(errors_p1_select)
            errors_p1_actionwise_select = torch.mean(errors_p1_select, dim=0)

            if cfg['DATASET']['Test']['P2']:
                errors_p2  = torch.stack(errors_p2)
                errors_p2_actionwise = torch.mean(errors_p2, dim=0)
                errors_p2_h  = torch.stack(errors_p2_h)
                errors_p2_actionwise_h = torch.mean(errors_p2_h, dim=0)
                errors_p2_mean  = torch.stack(errors_p2_mean)
                errors_p2_actionwise_mean = torch.mean(errors_p2_mean, dim=0)
                errors_p2_select  = torch.stack(errors_p2_select)
                errors_p2_actionwise_select = torch.mean(errors_p2_select, dim=0)

            if is_main_process():
                for ii in range(errors_p1_actionwise.shape[0]):
                    self.logger.info("step %d Protocol #1 (MPJPE) action-wise average J_Best: %.4f mm",
                            ii, errors_p1_actionwise[ii].item())
                    self.logger.info("step %d Protocol #1 (MPJPE) action-wise average P_Best: %.4f mm",
                            ii, errors_p1_actionwise_h[ii].item())
                    self.logger.info("step %d Protocol #1 (MPJPE) action-wise average P_Agg:  %.4f mm",
                            ii, errors_p1_actionwise_mean[ii].item())
                    self.logger.info("step %d Protocol #1 (MPJPE) action-wise average J_Agg:  %.4f mm",
                            ii, errors_p1_actionwise_select[ii].item())

                    if cfg['DATASET']['Test']['P2']:
                        self.logger.info("step %d Protocol #2 (MPJPE) action-wise average J_Best: %.4f mm",
                                ii, errors_p2_actionwise[ii].item())
                        self.logger.info("step %d Protocol #2 (MPJPE) action-wise average P_Best: %.4f mm",
                                ii, errors_p2_actionwise_h[ii].item())
                        self.logger.info("step %d Protocol #2 (MPJPE) action-wise average P_Agg:  %.4f mm",
                                ii, errors_p2_actionwise_mean[ii].item())
                        self.logger.info("step %d Protocol #2 (MPJPE) action-wise average J_Agg:  %.4f mm",
                                ii, errors_p2_actionwise_select[ii].item())

        # ---------- Normal aggregation (H36M etc.) ----------
        elif eval_type == 'Normal':
            if is_main_process():
                self.logger.info('Protocol #1   (MPJPE) action-wise average: %.1f mm',
                                torch.mean(torch.stack(errors_p1)).item())
                self.logger.info('Protocol #2 (P-MPJPE) action-wise average: %.1f mm',
                                torch.mean(torch.stack(errors_p2)).item())
                self.logger.info('Protocol #3 (N-MPJPE) action-wise average: %.1f mm',
                                torch.mean(torch.stack(errors_p3)).item())
                self.logger.info('Velocity      (MPJVE) action-wise average: %.2f mm',
                                torch.mean(torch.stack(errors_vel)).item())
import torch
from einops import rearrange, repeat
from remote_pdb import set_trace
from .eval_utils import *
import os
from modeling.dist import is_main_process

def eval_data_prepare(dataset_type, receptive_field, inputs_2d, inputs_3d):

    # ----------- 公共部分 -----------
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], \
        "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)

    inputs_2d_p = torch.squeeze(inputs_2d)   # [T,17,2]
    inputs_3d_p = torch.squeeze(inputs_3d)   # [T,17,3]
    T = inputs_2d_p.shape[0]

    # ============================================================
    #   A) seq2frame 模式：输入 RF 帧，输出中心帧（标准 PoseFormer）
    # ============================================================
    if dataset_type == "seq2frame":
        out_num = T - receptive_field + 1

        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
        eval_input_3d = torch.empty(out_num, 1, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

        for i in range(out_num):
            # 2D: sliding window
            eval_input_2d[i] = inputs_2d_p[i:i+receptive_field]

            # 3D: 中心帧监督
            center = i + receptive_field // 2
            eval_input_3d[i] = inputs_3d_p[center:center+1]

        return eval_input_2d, eval_input_3d


    # ============================================================
    #   B) seq2seq 模式：保持你原来的 chunked seq2seq（不动）
    # ============================================================
    else:
        # ---- 你的原始 seq2seq 逻辑，从这里开始完全不改 ----

        if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
            out_num = inputs_2d_p.shape[0] // receptive_field + 1
        else:
            out_num = inputs_2d_p.shape[0] // receptive_field

        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
        eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

        # 非重叠 chunk
        for i in range(out_num-1):
            eval_input_2d[i] = inputs_2d_p[i*receptive_field : i*receptive_field+receptive_field]
            eval_input_3d[i] = inputs_3d_p[i*receptive_field : i*receptive_field+receptive_field]

        # 长度不足 RF → pad
        if inputs_2d_p.shape[0] < receptive_field:
            from torch.nn import functional as F
            pad_right = receptive_field - inputs_2d_p.shape[0]
            inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
            inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
            inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')

        if inputs_3d_p.shape[0] < receptive_field:
            pad_right = receptive_field - inputs_3d_p.shape[0]
            inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
            inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
            inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')

        # 最后一个 clip → 尾部对齐
        eval_input_2d[-1] = inputs_2d_p[-receptive_field:]
        eval_input_3d[-1] = inputs_3d_p[-receptive_field:]

        return eval_input_2d, eval_input_3d


# ---------- DDHPOSE 输出与返回 ----------
def report_and_return_ddhpose(cfg, predicted_3d_pos_single, inputs_traj_single, inputs_3d_single, inputs_2d_single, cam, p1_dict, p2_dict, proj_func=None, cam_data=None):
    dataset_name = cfg['DATASET']['train_dataset']
    # 2d reprojection
    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz =predicted_3d_pos_single.shape
    
    inputs_traj_single_all = inputs_traj_single.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    predicted_3d_pos_abs_single = predicted_3d_pos_single + inputs_traj_single_all
    predicted_3d_pos_abs_single = predicted_3d_pos_abs_single.reshape(b_sz*t_sz*h_sz*f_sz, j_sz, c_sz) 
    cam_single_all = cam.repeat(b_sz*t_sz*h_sz*f_sz, 1)
    reproject_2d =proj_func(predicted_3d_pos_abs_single, cam_single_all)
    reproject_2d = reproject_2d.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, 2)
    
    # if dataset_name == '3DHP':
    #     target_2d = torch.from_numpy(image_coordinates(inputs_2d_single[..., :2].cpu().numpy(), w=cam_data[0], h=cam_data[1])).cuda()
    #     target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    
    error = mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single) # J-Best
    error_h = mpjpe_diffusion(predicted_3d_pos_single, inputs_3d_single) # P-Best
    error_mean = mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single, mean_pos=True) # P-Agg
    error_reproj_select = mpjpe_diffusion_reproj(predicted_3d_pos_single, inputs_3d_single, reproject_2d, inputs_2d_single) # J-Agg

    p1_dict['epoch_loss_3d_pos'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error.clone()
    p1_dict['epoch_loss_3d_pos_h'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_h.clone()
    p1_dict['epoch_loss_3d_pos_mean'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_mean.clone()
    p1_dict['epoch_loss_3d_pos_select'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error_reproj_select.clone()
    
    if cfg['DATASET']['Test']['P2']:
        error_p2 = p_mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single)
        error_h_p2 = p_mpjpe_diffusion(predicted_3d_pos_single, inputs_3d_single)
        error_mean_p2 = p_mpjpe_diffusion_all_min(predicted_3d_pos_single, inputs_3d_single, mean_pos=True)
        error_reproj_select_p2 = p_mpjpe_diffusion_reproj(predicted_3d_pos_single, inputs_3d_single, reproject_2d, inputs_2d_single)

        p2_dict['epoch_loss_3d_pos_p2'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.from_numpy(error_p2)
        p2_dict['epoch_loss_3d_pos_h_p2'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.from_numpy(error_h_p2)
        p2_dict['epoch_loss_3d_pos_mean_p2'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.from_numpy(error_mean_p2)
        p2_dict['epoch_loss_3d_pos_select_p2'] += inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.from_numpy(error_reproj_select_p2)


# ---------- MIXSTE 输出与返回 ----------
def report_and_return_mixste(cfg, predicted_3d_pos_single, inputs_3d_single, p1_dict):
    error = mpjpe(predicted_3d_pos_single, inputs_3d_single)
    
    p1_dict['epoch_loss_3d_pos_scale'] += (predicted_3d_pos_single.shape[0] * inputs_3d_single.shape[1] * n_mpjpe(predicted_3d_pos_single, inputs_3d_single))
    p1_dict['epoch_loss_3d_pos'] += (inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * error)

    inputs = inputs_3d_single.cpu().numpy().reshape(-1, inputs_3d_single.shape[-2], inputs_3d_single.shape[-1])
    predicted_3d_pos = predicted_3d_pos_single.cpu().numpy().reshape(-1, inputs_3d_single.shape[-2], inputs_3d_single.shape[-1])

    p1_dict['epoch_loss_3d_pos_procrustes'] += (inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.tensor(p_mpjpe(predicted_3d_pos, inputs)))
    p1_dict['epoch_loss_3d_vel'] += (inputs_3d_single.shape[0] * inputs_3d_single.shape[1] * torch.tensor(mean_velocity_error(predicted_3d_pos, inputs)))

    return p1_dict

@torch.no_grad()
def evaluate(cfg,
             test_loader,
             model_pos=None,
             kps_left=None,
             kps_right=None,
             joints_left=None,
             joints_right=None,
             action=None,
             logger=None):

    model_cfg = cfg['MODEL']
    eval_type = cfg["DATASET"]['Test']['Eval_type']
    test_name = cfg["DATASET"]['test_dataset']
    root_idx = cfg["DATASET"]["Root_idx"]
    quickdebug = cfg['DEBUG']

    # ---- accumulators for different eval types ----
    if eval_type == 'JPMA':
        # Hierarchical diffusion evaluation (DDHPose style)
        p1_dict = {
            'epoch_loss_3d_pos':        torch.zeros(model_cfg['backbone']['sampling_timesteps']).cuda(),
            'epoch_loss_3d_pos_h':      torch.zeros(model_cfg['backbone']['sampling_timesteps']).cuda(),
            'epoch_loss_3d_pos_mean':   torch.zeros(model_cfg['backbone']['sampling_timesteps']).cuda(),
            'epoch_loss_3d_pos_select': torch.zeros(model_cfg['backbone']['sampling_timesteps']).cuda(),
        }
        p2_dict = {
            'epoch_loss_3d_pos_p2':        torch.zeros(model_cfg['backbone']['sampling_timesteps']),
            'epoch_loss_3d_pos_h_p2':      torch.zeros(model_cfg['backbone']['sampling_timesteps']),
            'epoch_loss_3d_pos_mean_p2':   torch.zeros(model_cfg['backbone']['sampling_timesteps']),
            'epoch_loss_3d_pos_select_p2': torch.zeros(model_cfg['backbone']['sampling_timesteps']),
        }
    elif eval_type == 'Normal':
        # Normal single-hypothesis evaluation (MixSTE style)
        p1_dict = {
            'epoch_loss_3d_pos':            torch.zeros(1).cuda(),
            'epoch_loss_3d_pos_procrustes': torch.zeros(1).cuda(),
            'epoch_loss_3d_pos_scale':      torch.zeros(1).cuda(),
            'epoch_loss_3d_vel':            torch.zeros(1).cuda(),
        }
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    # Global counter used to normalize losses (in frames)
    N = 0

    # For 3DHP PCK/AUC（现在对 JPMA / Normal 都会算，只在 test_name == '3DHP' 时有效）
    sum_pck, sum_auc = 0.0, 0.0
    total_poses = 0

    with torch.no_grad():
        model_eval = model_pos
        model_eval.eval()
        model_eval.is_train = False

        if cfg["DATASET"]['train_dataset'] == cfg["DATASET"]['test_dataset']:

            for cam, batch_3d, batch_2d, seq_name, batch_act in test_loader:
                # -------- set projection function and camera meta --------
                if test_name == '3DHP':
                    if seq_name == "TS5" or seq_name == "TS6":
                        reproject_func = project_to_2d
                        cam_data = [2048, 2048, 10, 10]   # width, height, sensorSize_x, sensorSize_y
                    else:
                        reproject_func = project_to_2d_linear
                        cam_data = [1920, 1080, 10, 5.625]
                elif test_name == 'H36M':
                    reproject_func = project_to_2d
                    cam_data = None
                else:
                    reproject_func = None
                    cam_data = None

                # -------- squeeze and cast --------
                cam      = cam.squeeze(0)
                batch_3d = batch_3d.squeeze(0)
                batch_2d = batch_2d.squeeze(0)

                if cam is not None:
                    cam = cam.float()
                inputs_3d = batch_3d.float()
                inputs_2d = batch_2d.float()
                inputs_act = batch_act

                # -------- TTA: horizontal flip on 2D --------
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[..., 0] *= -1
                inputs_2d_flip[..., kps_left + kps_right, :] = inputs_2d_flip[..., kps_right + kps_left, :]

                # -------- prepare temporal sequences --------
                inputs_3d_p = inputs_3d
                inputs_2d, inputs_3d = eval_data_prepare(
                    cfg['DATASET']['dataset_type'],
                    cfg['DATASET']['receptive_field'],
                    inputs_2d, inputs_3d_p
                )
                inputs_2d_flip, _ = eval_data_prepare(
                    cfg['DATASET']['dataset_type'],
                    cfg['DATASET']['receptive_field'],
                    inputs_2d_flip, inputs_3d_p
                )

                if torch.cuda.is_available():
                    inputs_2d       = inputs_2d.cuda()
                    inputs_2d_flip  = inputs_2d_flip.cuda()
                    inputs_3d       = inputs_3d.cuda()
                    cam             = cam.cuda()

                # Root joint trajectory (only used in JPMA reprojection)
                inputs_traj = inputs_3d[..., root_idx:root_idx+1, :].clone()
                inputs_3d[..., root_idx, :] = 0

                # ======================= seq2seq setting =======================
                if cfg['DATASET']['dataset_type'] == 'seq2seq':
                    bs = cfg['DATASET']['batch_size']
                    total_batch = (inputs_3d.shape[0] + bs - 1) // bs

                    for batch_cnt in range(total_batch):
                        if (batch_cnt + 1) * bs > inputs_3d.shape[0]:
                            inputs_2d_single      = inputs_2d[batch_cnt * bs:]
                            inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:]
                            inputs_3d_single      = inputs_3d[batch_cnt * bs:]
                            inputs_traj_single    = inputs_traj[batch_cnt * bs:]
                        else:
                            inputs_2d_single      = inputs_2d[batch_cnt * bs:(batch_cnt + 1) * bs]
                            inputs_2d_flip_single = inputs_2d_flip[batch_cnt * bs:(batch_cnt + 1) * bs]
                            inputs_3d_single      = inputs_3d[batch_cnt * bs:(batch_cnt + 1) * bs]
                            inputs_traj_single    = inputs_traj[batch_cnt * bs:(batch_cnt + 1) * bs]

                        # ---- forward ----
                        predicted_3d_pos_single = model_eval(
                            inputs_2d=inputs_2d_single,
                            inputs_3d=inputs_3d_single,
                            input_2d_flip=inputs_2d_flip_single,
                            istrain=False,
                            inputs_act=inputs_act
                        )  # shape depends on eval_type / backbone

                        # zero root
                        predicted_3d_pos_single[..., root_idx, :] = 0

                        # ---- accumulate loss for different eval types ----
                        if eval_type == 'JPMA':
                            report_and_return_ddhpose(
                                cfg, predicted_3d_pos_single,
                                inputs_traj_single, inputs_3d_single,
                                inputs_2d_single, cam,
                                p1_dict, p2_dict,
                                proj_func=reproject_func, cam_data=cam_data
                            )
                        elif eval_type == 'Normal':
                            report_and_return_mixste(cfg, predicted_3d_pos_single, inputs_3d_single, p1_dict)

                        # ---- 3DHP PCK/AUC（JPMA / Normal 都算）----
                        if test_name == '3DHP':
                            J = inputs_3d_single.shape[-2]
                            C = inputs_3d_single.shape[-1]

                            # 对 JPMA 的多 hypothesis 输出，先在 H/T 维上做平均，得到单一预测
                            if predicted_3d_pos_single.dim() == 6:
                                # [B, T, H, F, J, C] -> mean over H, then T -> [B, F, J, C]
                                mean_pose = predicted_3d_pos_single.mean(dim=2)   # over H
                                mean_pose = mean_pose.mean(dim=1)                 # over T
                                pred_for_pck = mean_pose
                            else:
                                # Normal 情况，已经是 [B, F, J, C]（或类似），直接用
                                pred_for_pck = predicted_3d_pos_single

                            outputs_flat = pred_for_pck.contiguous().view(-1, J, C).cpu()
                            targets_flat = inputs_3d_single.contiguous().view(-1, J, C).cpu()
                            num_poses    = outputs_flat.shape[0]

                            pck = compute_PCK(targets_flat.numpy(), outputs_flat.numpy())
                            auc = compute_AUC(targets_flat.numpy(), outputs_flat.numpy())

                            sum_pck     += pck * num_poses
                            sum_auc     += auc * num_poses
                            total_poses += num_poses

                        # N counts number of (batch, frame) pairs consistently with loss
                        N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]

                        if quickdebug:
                            if N == inputs_3d_single.shape[0] * inputs_3d_single.shape[1]:
                                break

                # ======================= seq2frame setting =======================
                else:
                    predicted_3d_pos_single = model_eval(
                        inputs_2d=inputs_2d,
                        inputs_3d=inputs_3d,
                        input_2d_flip=inputs_2d_flip,
                        istrain=False,
                        inputs_act=inputs_act
                    )
                    predicted_3d_pos_single[..., root_idx, :] = 0

                    if eval_type == 'JPMA':
                        report_and_return_ddhpose(
                            cfg, predicted_3d_pos_single,
                            inputs_traj, inputs_3d,
                            inputs_2d_flip, cam,
                            p1_dict, p2_dict,
                            proj_func=reproject_func, cam_data=cam_data
                        )
                    elif eval_type == 'Normal':
                        report_and_return_mixste(cfg, predicted_3d_pos_single, inputs_3d, p1_dict)

                    if test_name == '3DHP':
                        J = inputs_3d.shape[-2]
                        C = inputs_3d.shape[-1]

                        if predicted_3d_pos_single.dim() == 6:
                            mean_pose = predicted_3d_pos_single.mean(dim=2)   # over H
                            mean_pose = mean_pose.mean(dim=1)                 # over T
                            pred_for_pck = mean_pose
                        else:
                            pred_for_pck = predicted_3d_pos_single

                        outputs_flat = pred_for_pck.contiguous().view(-1, J, C).cpu()
                        targets_flat = inputs_3d.contiguous().view(-1, J, C).cpu()
                        num_poses    = outputs_flat.shape[0]

                        pck = compute_PCK(targets_flat.numpy(), outputs_flat.numpy())
                        auc = compute_AUC(targets_flat.numpy(), outputs_flat.numpy())

                        sum_pck     += pck * num_poses
                        sum_auc     += auc * num_poses
                        total_poses += num_poses

                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    if quickdebug:
                        if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
                            break

            # After looping all sequences, finalize 3DHP PCK/AUC（JPMA / Normal 共用）
            if test_name == '3DHP' and total_poses > 0:
                epoch_pck = sum_pck / total_poses
                epoch_auc = sum_auc / total_poses
            else:
                epoch_pck, epoch_auc = None, None

        else:
            # If train_dataset != test_dataset, you can add another evaluation flow here if needed
            epoch_pck, epoch_auc = None, None

    # ---------------- finalize metrics & logging ----------------
    if is_main_process():
        if action is None:
            logger.info("----------")
        else:
            logger.info("----%s----", action)

    # =====================================================
    # JPMA branch
    # =====================================================
    if eval_type == 'JPMA':
        e1        = (p1_dict['epoch_loss_3d_pos']        / N) * 1000
        e1_h      = (p1_dict['epoch_loss_3d_pos_h']      / N) * 1000
        e1_mean   = (p1_dict['epoch_loss_3d_pos_mean']   / N) * 1000
        e1_select = (p1_dict['epoch_loss_3d_pos_select'] / N) * 1000

        if cfg['DATASET']['Test']['P2']:
            e2        = (p2_dict['epoch_loss_3d_pos_p2']        / N) * 1000
            e2_h      = (p2_dict['epoch_loss_3d_pos_h_p2']      / N) * 1000
            e2_mean   = (p2_dict['epoch_loss_3d_pos_mean_p2']   / N) * 1000
            e2_select = (p2_dict['epoch_loss_3d_pos_select_p2'] / N) * 1000
        else:
            e2 = e2_h = e2_mean = e2_select = None

        if is_main_process():
            for ii in range(e1.shape[0]):
                logger.info("step %d : Protocol #1 Error (MPJPE) J_Best:  %.4f mm", ii, e1[ii].item())
                logger.info("step %d : Protocol #1 Error (MPJPE) P_Best:  %.4f mm", ii, e1_h[ii].item())
                logger.info("step %d : Protocol #1 Error (MPJPE) P_Agg:   %.4f mm", ii, e1_mean[ii].item())
                logger.info("step %d : Protocol #1 Error (MPJPE) J_Agg:   %.4f mm", ii, e1_select[ii].item())

            if test_name == '3DHP' and epoch_pck is not None and epoch_auc is not None:
                logger.info("3DHP PCK (JPMA overall): %.2f",  epoch_pck)
                logger.info("3DHP AUC (JPMA overall): %.4f", epoch_auc)

            logger.info("----------")

        # 返回值：3DHP 比普通多两个（PCK/AUC），其它数据集保持原接口
        if test_name == '3DHP':
            if cfg['DATASET']['Test']['P2']:
                return e1, e1_h, e1_mean, e1_select, e2, e2_h, e2_mean, e2_select, epoch_pck, epoch_auc
            else:
                return e1, e1_h, e1_mean, e1_select, epoch_pck, epoch_auc
        else:
            if cfg['DATASET']['Test']['P2']:
                return e1, e1_h, e1_mean, e1_select, e2, e2_h, e2_mean, e2_select
            else:
                return e1, e1_h, e1_mean, e1_select

    # =====================================================
    # Normal branch
    # =====================================================
    elif eval_type == 'Normal':
        e1 = (p1_dict['epoch_loss_3d_pos']           / N) * 1000
        e2 = (p1_dict['epoch_loss_3d_pos_procrustes'] / N) * 1000
        e3 = (p1_dict['epoch_loss_3d_pos_scale']    / N) * 1000
        ev = (p1_dict['epoch_loss_3d_vel']          / N) * 1000

        if is_main_process():
            logger.info("Protocol #1 Error (MPJPE):   %.4f mm", e1)
            logger.info("Protocol #2 Error (P-MPJPE): %.4f mm", e2)
            logger.info("Protocol #3 Error (N-MPJPE): %.4f mm", e3)
            logger.info("Velocity Error (MPJVE):      %.4f mm", ev)
            if test_name == '3DHP' and epoch_pck is not None and epoch_auc is not None:
                logger.info("3DHP PCK (overall):         %.2f",  epoch_pck)
                logger.info("3DHP AUC (overall):         %.4f", epoch_auc)
            logger.info("----------")

        # 对非 3DHP，epoch_pck/epoch_auc 会是 None，保持兼容
        return e1, e2, e3, ev, epoch_pck, epoch_auc
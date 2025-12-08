import torch
from einops import rearrange, repeat
from remote_pdb import set_trace
from .eval_utils import *
import os
from modeling.dist import is_main_process
from scipy.io import savemat

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

def select_ddhpose_multi_hypothesis(
    predicted_3d_pos_single: torch.Tensor,
    inputs_3d_single: torch.Tensor,
    inputs_2d_single: torch.Tensor,
    reproject_2d: torch.Tensor,
    cam_data=None,
):
    """
    Select four representative poses from multi-hypothesis DDHPose outputs.

    Args:
        predicted_3d_pos_single: [B, T, H, F, J, C]
        inputs_3d_single:        [B, F, J, C]
        inputs_2d_single:        [B, F, J, 2 or C>=2]
        reproject_2d:            [B, T, H, F, J, 2]
        cam_data:                (W, H, sensor_x, sensor_y) or None

    Returns:
        mean_pose:      [B, T, F, J, C]  # P-Agg
        h_min_pose:     [B, T, F, J, C]  # P-Best
        joint_min_pose: [B, T, F, J, C]  # J-Best
        reproj_min_pose:[B, T, F, J, C]  # J-Agg
    """
    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted_3d_pos_single.shape

    # 1) P-Agg: mean over head dimension
    mean_pose = predicted_3d_pos_single.mean(dim=2, keepdim=False)  # [B, T, F, J, C]

    # 2) P-Best: choose the head with minimal pose-wise MPJPE
    target = inputs_3d_single.unsqueeze(1).unsqueeze(1)             # [B, 1, 1, F, J, C]
    target = target.repeat(1, t_sz, h_sz, 1, 1, 1)                  # [B, T, H, F, J, C]
    errors = torch.norm(predicted_3d_pos_single - target, dim=-1)   # [B, T, H, F, J]

    # average over (F, J) to get error per (T, H)
    errors_h = rearrange(errors, 'b t h f n -> t h b f n').reshape(t_sz, h_sz, -1)  # [T, H, B*F*J]
    errors_h = errors_h.mean(dim=-1, keepdim=True)                  # [T, H, 1]
    h_min_indices = torch.min(errors_h, dim=1, keepdim=True).indices  # [T, 1, 1]
    h_min_indices = h_min_indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)         # [1, T, 1, 1, 1, 1]
    h_min_indices = h_min_indices.repeat(b_sz, 1, 1, f_sz, j_sz, c_sz)             # [B, T, 1, F, J, C]
    h_min_pose = torch.gather(predicted_3d_pos_single, 2, h_min_indices).squeeze(2)  # [B, T, F, J, C]

    # 3) J-Best: choose best head per joint
    joint_min_indices = torch.min(errors, dim=2, keepdim=True).indices              # [B, T, 1, F, J]
    joint_min_indices = joint_min_indices.unsqueeze(-1).repeat(1, 1, 1, 1, 1, c_sz) # [B, T, 1, F, J, C]
    joint_min_pose = torch.gather(predicted_3d_pos_single, 2, joint_min_indices).squeeze(2)  # [B, T, F, J, C]

    # 4) J-Agg: reprojection-based head selection
    if cam_data is not None:
        target_2d_np = image_coordinates(
            inputs_2d_single[..., :2].cpu().numpy(),
            w=cam_data[0], h=cam_data[1]
        )
        target_2d = torch.from_numpy(target_2d_np).to(reproject_2d.device)
    else:
        target_2d = inputs_2d_single[..., :2].to(reproject_2d.device)

    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)    # [B, T, H, F, J, 2]
    errors_2d = torch.norm(reproject_2d - target_2d, dim=-1)                          # [B, T, H, F, J]
    reproj_min_indices = torch.min(errors_2d, dim=2, keepdim=True).indices            # [B, T, 1, F, J]
    reproj_min_indices = reproj_min_indices.unsqueeze(-1).repeat(1, 1, 1, 1, 1, c_sz) # [B, T, 1, F, J, C]
    reproj_min_pose = torch.gather(predicted_3d_pos_single, 2, reproj_min_indices).squeeze(2)  # [B, T, F, J, C]

    return mean_pose, h_min_pose, joint_min_pose, reproj_min_pose

def pose_post_process(pose_pred, data_list, keys, receptive_field):
    # normalize keys: ('TS1',) / ['TS1'] -> 'TS1'
    if isinstance(keys, (tuple, list)):
        keys = keys[0]
    for ii in range(pose_pred.shape[0] - 1):
        data_list[keys][:, ii * receptive_field:(ii + 1) * receptive_field] = pose_pred[ii]
    data_list[keys][:, -receptive_field:] = pose_pred[-1]
    return data_list
# ---------- DDHPOSE 输出与返回 ----------
def report_and_return_ddhpose(cfg, predicted_3d_pos_single, inputs_traj_single, inputs_3d_single, inputs_2d_single, \
                              cam, p1_dict, p2_dict, proj_func=None, cam_data=None, return_inference = False):
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
    
    if not return_inference:
        return None, None, None, None
    mean_pose, h_min_pose, joint_min_pose, reproj_min_pose = select_ddhpose_multi_hypothesis(
        predicted_3d_pos_single,
        inputs_3d_single,
        inputs_2d_single,
        reproject_2d,
        cam_data=cam_data,
    )
    return mean_pose, h_min_pose, joint_min_pose, reproj_min_pose
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
    save_mat = (test_name == '3DHP' and eval_type == 'JPMA')
    if save_mat:
        # dict: seq_name -> np.ndarray (after pose_post_process)
        data_inference_mean      = {}  # P_Agg
        data_inference_h_min     = {}  # P_Best
        data_inference_joint_min = {}  # J_Best
        data_inference_reproj_min= {}  # J_Agg

        # buffer: seq_name -> lists of chunk-wise predictions
        seq_buffers = {}

    # For 3DHP PCK/AUC（现在对 JPMA / Normal 都会算，只在 test_name == '3DHP' 时有效）
    sum_pck, sum_auc = 0.0, 0.0
    total_poses = 0

    with torch.no_grad():
        model_eval = model_pos
        model_eval.eval()
        model_eval.is_train = False

        if cfg["DATASET"]['train_dataset'] == cfg["DATASET"]['test_dataset']:

            for cam, batch_3d, batch_2d, seq_name, batch_act in test_loader:

                if isinstance(seq_name, (list, tuple)):
                    seq_key = seq_name[0]
                else:
                    seq_key = seq_name
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

                # -------- init buffer for this sequence (if saving) --------
                if save_mat and seq_key not in seq_buffers:
                    seq_buffers[seq_key] = {
                        "mean":      [],
                        "h_min":     [],
                        "joint_min": [],
                        "reproj":    [],
                    }
                    if seq_key not in data_inference_mean:
                        T    = model_cfg['backbone']['sampling_timesteps']
                        _, f_sz, j_sz, c_sz = inputs_3d.shape  # full sequence length BEFORE chunking

                        data_inference_mean[seq_key]       = np.zeros((T, f_sz, j_sz, c_sz), dtype=np.float32)
                        data_inference_h_min[seq_key]      = np.zeros((T, f_sz, j_sz, c_sz), dtype=np.float32)
                        data_inference_joint_min[seq_key]  = np.zeros((T, f_sz, j_sz, c_sz), dtype=np.float32)
                        data_inference_reproj_min[seq_key] = np.zeros((T, f_sz, j_sz, c_sz), dtype=np.float32)
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

                        if eval_type == 'JPMA':
                            mean_pose, h_min_pose, joint_min_pose, reproj_min_pose = report_and_return_ddhpose(
                                cfg,
                                predicted_3d_pos_single,
                                inputs_traj_single,
                                inputs_3d_single,
                                inputs_2d_single,
                                cam,
                                p1_dict,
                                p2_dict,
                                proj_func=reproject_func,
                                cam_data=cam_data,
                                return_inference=save_mat,
                            )

                            # 3DHP + JPMA: cache pose for saving .mat
                            if save_mat:
                                buf = seq_buffers[seq_key]
                                # each: [B, T, F, J, C]
                                buf["mean"].append(mean_pose.cpu().numpy() * 1000.0)
                                buf["h_min"].append(h_min_pose.cpu().numpy() * 1000.0)
                                buf["joint_min"].append(joint_min_pose.cpu().numpy() * 1000.0)
                                buf["reproj"].append(reproj_min_pose.cpu().numpy() * 1000.0)

                        elif eval_type == 'Normal':
                            report_and_return_mixste(cfg, predicted_3d_pos_single, inputs_3d_single, p1_dict)

                        if quickdebug:
                            break
                        N += inputs_3d_single.shape[0] * inputs_3d_single.shape[1]

                    # -------- after all chunks of this sequence: do pose_post_process (3DHP+JPMA only) --------
                    if save_mat:
                        scale = 1000.0
                        buf = seq_buffers[seq_key]
                        # concatenate along chunk dimension: [num_chunks, T, F_win, J, C]
                        mean_np   = np.concatenate(buf["mean"],      axis=0)
                        hmin_np   = np.concatenate(buf["h_min"],     axis=0)
                        jbest_np  = np.concatenate(buf["joint_min"], axis=0)
                        reproj_np = np.concatenate(buf["reproj"],    axis=0)

                        data_inference_mean      = pose_post_process(mean_np,   data_inference_mean,      seq_name, cfg['DATASET']['receptive_field'])
                        data_inference_h_min     = pose_post_process(hmin_np,   data_inference_h_min,     seq_name, cfg['DATASET']['receptive_field'])
                        data_inference_joint_min = pose_post_process(jbest_np,  data_inference_joint_min, seq_name, cfg['DATASET']['receptive_field'])
                        data_inference_reproj_min= pose_post_process(reproj_np, data_inference_reproj_min,seq_name, cfg['DATASET']['receptive_field'])

                    if quickdebug:
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
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    if quickdebug:
                        if N == inputs_3d.shape[0] * inputs_3d.shape[1]:
                            break

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

            if cfg['DATASET']['Test']['P2']:
                for ii in range(e2.shape[0]):
                    logger.info("step %d : Protocol #2 Error (P-MPJPE) J_Best:  %.4f mm", ii, e2[ii].item())
                    logger.info("step %d : Protocol #2 Error (P-MPJPE) P_Best:  %.4f mm", ii, e2_h[ii].item())
                    logger.info("step %d : Protocol #2 Error (P-MPJPE) P_Agg:   %.4f mm", ii, e2_mean[ii].item())
                    logger.info("step %d : Protocol #2 Error (P-MPJPE) J_Agg:   %.4f mm", ii, e2_select[ii].item())
                    
            # if test_name == '3DHP' and epoch_pck is not None and epoch_auc is not None:
            #     logger.info("3DHP PCK (JPMA overall): %.2f",  epoch_pck)
            #     logger.info("3DHP AUC (JPMA overall): %.4f", epoch_auc)

            logger.info("----------")

        # save 3DHP inference .mat files
        if save_mat and is_main_process():
            save_dir = os.path.join("outputs", cfg["DATASET"]["test_dataset"], cfg["MODEL"]["name"], cfg["ENGINE"]["save_name"])
            mat_path_mean      = os.path.join(save_dir, 'inference_data_P_Agg.mat')
            mat_path_h_min     = os.path.join(save_dir, 'inference_data_P_Best.mat')
            mat_path_joint_min = os.path.join(save_dir, 'inference_data_J_Best.mat')
            mat_path_reproj_min= os.path.join(save_dir, 'inference_data_J_Agg.mat')

            savemat(mat_path_mean,      data_inference_mean)
            savemat(mat_path_h_min,     data_inference_h_min)
            savemat(mat_path_joint_min, data_inference_joint_min)
            savemat(mat_path_reproj_min,data_inference_reproj_min)

            logger.info("Saved 3DHP JPMA inference mats to:")
            logger.info("  %s", mat_path_mean)
            logger.info("  %s", mat_path_h_min)
            logger.info("  %s", mat_path_joint_min)
            logger.info("  %s", mat_path_reproj_min)


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
            logger.info("Protocol #1 Error (MPJPE):   %.4f mm", e1.item())
            logger.info("Protocol #2 Error (P-MPJPE): %.4f mm", e2.item())
            logger.info("Protocol #3 Error (N-MPJPE): %.4f mm", e3.item())
            logger.info("Velocity Error (MPJVE):      %.4f mm", ev.item())
            # if test_name == '3DHP' and epoch_pck is not None and epoch_auc is not None:
            #     logger.info("3DHP PCK (overall):         %.2f",  epoch_pck)
            #     logger.info("3DHP AUC (overall):         %.4f", epoch_auc)
            logger.info("----------")

        return e1, e2, e3, ev
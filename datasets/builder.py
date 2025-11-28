# ddhpose/datasets/builder.py
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ===== 你项目里的真实路径：按需修改 =====
from .camera import world_to_camera, normalize_screen_coordinates
from .h36m_dataset import Human36mDataset   
from .threedhp_dataset import *
# =======================================
from remote_pdb import set_trace
from .utils import *
from .generators_h36m import *
from .generators_3dhp import *
import torch.distributed as dist
from .collate_fn import collate_keep_seqname

@dataclass
class Bundle:
    train_loader: Optional[DataLoader] = None
    val_loader:   Optional[DataLoader] = None
    test_loader:  Optional[DataLoader] = None
    dataset:      Optional[object]     = None   # 原始 dataset 对象（可选）
    keypoints_2d: Optional[dict]       = None
    kps_left:     Optional[List[int]]  = None
    kps_right:    Optional[List[int]]  = None
    joints_left:  Optional[List[int]]  = None
    joints_right: Optional[List[int]]  = None
    action_key:  Optional[str]        = ""     # 测试时该 Bundle 对应的动作
    action_filter: Optional[List[str]] = None  # 测试时该 Bundle 对应的动作过滤列表
    bone_index:   Optional[List[Tuple[int,int]]] = None  # 可按需填充

# ---------------- 基础加载（来自你旧 main） ----------------
def _load_dataset(cfg) -> object:
    name = cfg["DATASET"]["train_dataset"].lower()
    root = cfg["DATASET"].get("root", "data")
    
    if name == "h36m" or name == "human3.6m":
        path_3d = os.path.join(root, f"data_3d_{name}.npz")
        dataset = Human36mDataset(path_3d, cfg["DATASET"])
        # 准备 3D：world->camera 并根对齐
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                if "positions" in anim:
                    positions_3d = []
                    for cam in anim["cameras"]:
                        pos_3d = world_to_camera(anim["positions"], R=cam["orientation"], t=cam["translation"])
                        pos_3d[:, 1:] -= pos_3d[:, :1]
                        positions_3d.append(pos_3d)
                    anim["positions_3d"] = positions_3d
    elif name == "3dhp":
        dataset = ThreeDHPDataset(root, name)
        data_train = dataset.data_train
        data_test = dataset.data_test
        joints_left, kps_right = dataset.joints_left, dataset.joints_right
        # return data_train, data_test, joints_left, kps_right
        return dataset
    else:
        raise KeyError(f"Unsupported DATASET.name: {name}")


    return dataset

def _load_keypoints_3DHP(data_train, data_test):
    out_poses_3d_train = {}
    out_poses_2d_train = {}
    out_poses_3d_test = {}
    out_poses_2d_test = {}
    valid_frame = {}
    for seq in data_train.keys():
        for cam in data_train[seq][0].keys():
            anim = data_train[seq][0][cam]

            subject_name, seq_name = seq.split(" ")

            data_3d = anim['data_3d']
            data_3d[:, :14] -= data_3d[:, 14:15]
            data_3d[:, 15:] -= data_3d[:, 14:15]
            out_poses_3d_train[(subject_name, seq_name, cam)] = data_3d

            data_2d = anim['data_2d']
            data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
            out_poses_2d_train[(subject_name, seq_name, cam)] = data_2d

    for seq in data_test.keys():

        anim = data_test[seq]

        valid_frame[seq] = anim["valid"]

        data_3d = anim['data_3d']
        data_3d[:, :14] -= data_3d[:, 14:15]
        data_3d[:, 15:] -= data_3d[:, 14:15]
        out_poses_3d_test[seq] = data_3d

        data_2d = anim['data_2d']
        
        if seq == "TS5" or seq == "TS6":
            width = 1920
            height = 1080
        else:
            width = 2048
            height = 2048
        data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
        out_poses_2d_test[seq] = data_2d

    return out_poses_3d_train, out_poses_2d_train, out_poses_3d_test, out_poses_2d_test

def _load_keypoints_2d(cfg, dataset) -> Tuple[dict, dict, List[int], List[int], List[int], List[int]]:
    name = cfg["DATASET"]["train_dataset"].lower()
    root = cfg["DATASET"].get("root", "data")
    kp_tag = cfg["DATASET"].get("keypoints", "cpn_ft_h36m_dbb")
    if not cfg["DATASET"]["Cross_Dataset"]:
        path_2d = os.path.join(root, f"data_2d_{name}_{kp_tag}.npz")
        keypoints = np.load(path_2d, allow_pickle=True)
        meta      = keypoints["metadata"].item()
        sym       = meta["keypoints_symmetry"]
        kps_left, kps_right = list(sym[0]), list(sym[1])
        joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    else:
        path_2d = os.path.join(root, "h36m_16joints" , f"data_2d_{name}_{kp_tag}.npz")
        keypoints = np.load(path_2d, allow_pickle=True)
        joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        kps_left, kps_right = joints_left, joints_right

    keypoints_2d = keypoints["positions_2d"].item()

    return keypoints_2d, kps_left, kps_right, joints_left, joints_right

def _align_kp_and_mocap_lengths(dataset, keypoints_2d):
    for subject in dataset.subjects():
        assert subject in keypoints_2d, f"Subject {subject} missing in 2D detections"
        for action in dataset[subject].keys():
            assert action in keypoints_2d[subject], f"Action {action} of subject {subject} missing in 2D detections"
            if "positions_3d" not in dataset[subject][action]:
                continue
            for cam_idx in range(len(keypoints_2d[subject][action])):
                mocap_len = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                assert keypoints_2d[subject][action][cam_idx].shape[0] >= mocap_len
                if keypoints_2d[subject][action][cam_idx].shape[0] > mocap_len:
                    keypoints_2d[subject][action][cam_idx] = keypoints_2d[subject][action][cam_idx][:mocap_len]
            assert len(keypoints_2d[subject][action]) == len(dataset[subject][action]["positions_3d"])

def _normalize_keypoints(dataset, keypoints_2d):
    for subject in keypoints_2d.keys():
        for action in keypoints_2d[subject]:
            for cam_idx, kps in enumerate(keypoints_2d[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam["res_w"], h=cam["res_h"])
                keypoints_2d[subject][action][cam_idx] = kps

def _build_splits(cfg, render_mode: bool = False, viz_subject: str = ""):
    ds = cfg["DATASET"]
    subjects_train = ds["subjects_train"].split(",")
    subjects_semi = [] if (ds["subjects_unlabeled"]=="") else ds["subjects_unlabeled"].split(',')
    if not render_mode:
        subjects_test = ds["subjects_test"].split(',')
    else:
        subjects_test = [viz_subject]
    return subjects_train, subjects_semi, subjects_test

def get_all_actions_by_subject(cfg, dataset, subjects_test):
    all_actions = {}
    all_actions_flatten = []
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_flatten.append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))
    action_filter = None if cfg['DATASET']['Test']['actions'] == '*' else cfg['DATASET']['Test']['actions'].split(',')
    
    return all_actions, action_filter

# ---------------- 数据展开与窗口化 ----------------
def fetch(dataset, keypoints_2d, subjects, action_filter=None, subset=1.0, parse_3d_poses=True, downsample=1):
    out_poses_3d, out_poses_2d, out_camera_params, out_action = [], [], [], []

    for subject in subjects:
        for action in keypoints_2d[subject].keys():
            if action_filter is not None:
                if not any(action.startswith(a) for a in action_filter):
                    continue

            poses_2d = keypoints_2d[subject][action]
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])
                out_action.append(action)

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample

    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_total = len(out_poses_2d[i])
            n_frames = int(round(n_total // stride * subset) * stride)
            start = deterministic_random(0, n_total - n_frames + 1, str(n_total))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d, out_action

def fetch_actions(actions, keypoints, dataset, downsample=1):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)): # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

        if subject in dataset.cameras():
            cams = dataset.cameras()[subject]
            assert len(cams) == len(poses_2d), 'Camera count mismatch'
            for cam in cams:
                if 'intrinsic' in cam:
                    out_camera_params.append(cam['intrinsic'])

    stride = downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d

def build_data_bundle_test_H36M(cfg, kps_left, kps_right, joints_left, joints_right, dataset, keypoints, actions, action_filter=None, training=False, PoseUnchunkedDataset=None):
    test_bundle_list = []
    for action_key in actions.keys():
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue

        cameras_act, poses_act, poses_2d_act = fetch_actions(actions[action_key], keypoints, dataset)
        act_dataset = PoseUnchunkedDataset(poses_2d_act, poses_act, cameras_act,
                                        pad=(cfg["DATASET"]["receptive_field"] - cfg["DATASET"]["chunk_size"]) // 2, 
                                        causal_shift=0, augment=True,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right)

        test_loader = DataLoader(act_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_keep_seqname)
        test_bundle_list.append(
            Bundle(
                test_loader=test_loader,
                dataset=dataset,
                keypoints_2d=keypoints,
                kps_left=kps_left, kps_right=kps_right,
                joints_left=joints_left, joints_right=joints_right,
                action_key=action_key,
                action_filter=action_filter,
                bone_index=cfg.get("DATASET", {}).get("bone_index", None)  # 可在 cfg 里定义
            )
        )
    return test_bundle_list

def build_data_bundle_test_3DHP(cfg, kps_left, kps_right, joints_left, joints_right, poses_valid, poses_valid_2d, actions, action_filter=None, training=False, PoseUnchunkedDataset=None):
    test_bundle_list = []
    for action_key in poses_valid.keys():
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue
       
        act_dataset = PoseUnchunkedDataset(poses_valid_2d, poses_valid, None,
                                        pad=(cfg["DATASET"]["number_of_frames"] -1) // 2, 
                                        causal_shift=0, augment=True,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right)
        
        test_loader = DataLoader(act_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_bundle_list.append(
            Bundle(
                test_loader=test_loader,
                dataset=act_dataset,
                # keypoints_2d=keypoints,
                kps_left=kps_left, kps_right=kps_right,
                joints_left=joints_left, joints_right=joints_right,
                action_key=action_key,
                action_filter=action_filter,
                bone_index=cfg.get("DATASET", {}).get("bone_index", None)  # 可在 cfg 里定义
            )
        )
    return test_bundle_list

def build_data_bundle(cfg, training: bool = True) -> Bundle:
    train_dataset = cfg["DATASET"]["train_dataset"]
    test_dataset = cfg["DATASET"]["test_dataset"]
    # 拉平成序列列表
    ds_cfg = cfg["DATASET"]
    subset       = float(ds_cfg.get("subset", 1.0))
    downsample   = int(ds_cfg.get("downsample", 1))   # 全局下采样
    T            = int(ds_cfg.get("seq_len", 243))
    # 组装 Dataset & DataLoader
    # 注意：OpenGait 风格是 DDP 分布式采样器
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    receptive_field = ds_cfg["receptive_field"]
    batch_size = ds_cfg["batch_size"]
    stride = ds_cfg["stride"]
    causal_shift = 0

    chunk_cls_name = f"PoseChunkDataset_{train_dataset}"
    unchunk_cls_name = f"PoseUnchunkedDataset_{train_dataset}"
    try:
        ChunkDataset = eval(chunk_cls_name)
        UnchunkDataset = eval(unchunk_cls_name)
    except NameError as e:
        raise ValueError(f"找不到对应的数据集类: {e}")
    
    # 读取 + 预处理
    if training:
        dataset = _load_dataset(cfg)
        if train_dataset == "H36M":
            keypoints_2d, kps_left, kps_right, joints_left, joints_right = _load_keypoints_2d(cfg, dataset)

            # 对齐长度 & 归一化
            if any("positions_3d" in dataset[s][a] for s in dataset.subjects() for a in dataset[s].keys()):
                _align_kp_and_mocap_lengths(dataset, keypoints_2d)
            _normalize_keypoints(dataset, keypoints_2d)

            # 划分 subject
            subjects_train, subjects_semi, subjects_test = _build_splits(cfg)  

            cameras_train, poses_train, poses_train_2d, action_train = fetch(dataset, keypoints_2d, subjects_train, subset=subset, downsample=downsample)
            cameras_valid, poses_valid, poses_valid_2d, action_valid = fetch(dataset, keypoints_2d, subjects_test,  subset=1.0, downsample=downsample)

        elif train_dataset == "3DHP":
            # data_train, data_test, kps_left, kps_right = _load_dataset(cfg)
            data_train, data_test, kps_left, kps_right = dataset.data_train, dataset.data_test, dataset.kps_left, dataset.kps_right
            joints_left, joints_right = kps_left, kps_right
            poses_train, poses_train_2d, poses_valid, poses_valid_2d   = _load_keypoints_3DHP(data_train, data_test)
            cameras_train, cameras_valid = None, None

    
        train_dataset = ChunkDataset(poses_train_2d, poses_train, cameras_train, action_train,
                                chunk_length=ds_cfg["chunk_size"],
                                pad= (receptive_field - ds_cfg["chunk_size"]) // 2, 
                                causal_shift=causal_shift,
                                augment=True,
                                kps_left=kps_left, kps_right=kps_right,
                                joints_left=joints_left, joints_right=joints_right)
        sampler = DistributedSampler(train_dataset, num_replicas=torch.distributed.get_world_size(), rank=dist.get_rank(), shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size//stride, sampler=sampler, num_workers=4, pin_memory=True)
    
        # 验证集简单用测试 subject 的较稀疏滑窗
        val_dataset = UnchunkDataset(poses_valid_2d, poses_valid, cameras_valid, action_valid,
                pad=(receptive_field -1) // 2, 
                causal_shift=causal_shift,
                augment=False,
                kps_left=kps_left, kps_right=kps_right,
                joints_left=joints_left, joints_right=joints_right)

        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,      # 可以比训练大，比如 4/8/16，看显存
            sampler=val_sampler,         # 关键：恢复分片
            shuffle=False,
            num_workers=2,                # eval 往往 GPU-bound，过大反而有IPC开销
            pin_memory=True,
            persistent_workers=True,       # PyTorch>=1.8，避免每轮重启
            collate_fn=collate_keep_seqname
        )
        return Bundle(
            train_loader=train_loader,
            val_loader=val_loader,
            dataset=dataset,
            # keypoints_2d=keypoints_2d,
            kps_left=kps_left, kps_right=kps_right,
            joints_left=joints_left, joints_right=joints_right,
            bone_index=cfg.get("DATASET", {}).get("bone_index", None)  # 可在 cfg 里定义
        )
    else:
        if train_dataset == test_dataset:
            dataset = _load_dataset(cfg)
            keypoints_2d, kps_left, kps_right, joints_left, joints_right = _load_keypoints_2d(cfg, dataset)

            # 对齐长度 & 归一化
            if any("positions_3d" in dataset[s][a] for s in dataset.subjects() for a in dataset[s].keys()):
                _align_kp_and_mocap_lengths(dataset, keypoints_2d)
            _normalize_keypoints(dataset, keypoints_2d)

            # 划分 subject
            subjects_train, subjects_semi, subjects_test = _build_splits(cfg)  
            
            build_data_bundle_test = eval(f"build_data_bundle_test_{test_dataset}")
            if test_dataset == "H36M":
                all_actions, action_filter = get_all_actions_by_subject(cfg, dataset, subjects_test)
                bundle_list = build_data_bundle_test(cfg, kps_left, kps_right, joints_left, joints_right, dataset, keypoints_2d, \
                                                    all_actions, action_filter, PoseUnchunkedDataset=UnchunkDataset)
            elif test_dataset == "3DHP":
                all_actions, action_filter = {}, None
                bundle_list = build_data_bundle_test(cfg, kps_left, kps_right, joints_left, joints_right, poses_valid, poses_valid_2d, \
                                                    all_actions, action_filter, PoseUnchunkedDataset=UnchunkDataset)
            return bundle_list

        else:
            ############################################
            # prepare cross dataset validation
            ############################################
            test_bundle_list = []
            if test_dataset == '3DHP':
                mpi3d_npz = np.load('data/test_set/test_{:}.npz'.format(test_dataset.lower()))  
                tmp = mpi3d_npz
                dataset = PoseBuffer([tmp['pose3d']], [tmp['pose2d']])
                mpi3d_loader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=False, num_workers=4, pin_memory=True)         
            kps_left, kps_right = dataset.kps_left, dataset.kps_right
            joints_left, joints_right = dataset.joints_left, dataset.joints_right

            test_bundle_list.append(
                Bundle(
                    test_loader=mpi3d_loader,
                    dataset=dataset,
                    # keypoints_2d=keypoints,
                    kps_left=kps_left, kps_right=kps_right,
                    joints_left=joints_left, joints_right=joints_right,
                    # action_key=action_key,
                    # action_filter=action_filter,
                    bone_index=cfg.get("DATASET", {}).get("bone_index", None)  # 可在 cfg 里定义
                )
            )
            return test_bundle_list
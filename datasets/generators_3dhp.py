# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
from torch.utils.data import Dataset
from remote_pdb import set_trace
import torch

def cam_mm_to_pix(cam, cam_data):
    # w, h, ss_x, ss_y
    mx = cam_data[0] / cam_data[2]
    my = cam_data[1] / cam_data[3]
    cam[0] = cam[0] * mx
    cam[1] = cam[1] * my
    cam[2] = cam[2] * mx + cam_data[0]/2
    cam[3] = cam[3] * my + cam_data[1]/2

    return cam

class PoseChunkDataset_3DHP(Dataset):
    def __init__(self, poses_2d, poses_3d=None, cameras=None, action_train=None,
                 chunk_length=1, pad=0, causal_shift=0, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None,
                 joints_left=None, joints_right=None,
                 dataset_type='seq2frame', frame_stride=1, tds=1):

        """
        poses_2d / poses_3d / cameras: dict, key = (subject, seq, cam_index)
            poses_2d[key]: [T, J, 2]
            poses_3d[key]: [T, J, 3]
            cameras[key]:  camera params for this sequence
        """

        self.poses_2d = poses_2d
        self.poses_3d = poses_3d
        self.cameras = cameras

        self.chunk_length = chunk_length
        self.pad = pad
        self.causal_shift = causal_shift
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.random = np.random.RandomState(random_seed)

        self.dataset_type = dataset_type  # 'seq2seq' or 'seq2frame'
        self.frame_stride = frame_stride
        self.tds = tds

        # ----------------- build pairs: (seq_key, start, end, flip) -----------------
        self.pairs = []

        for key in self.poses_2d.keys():
            # key is (subject, seq, cam_index)
            assert (self.poses_3d is None
                    or self.poses_2d[key].shape[0] == self.poses_3d[key].shape[0])

            n_frames = self.poses_2d[key].shape[0]
            n_chunks = (n_frames + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - n_frames) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset  # [0..n_chunks]*L - offset

            # seq_key 用 numpy array 存三元组，保持和你原版一致
            keys = np.tile(np.array(key).reshape(1, 3), (len(bounds) - 1, 1))

            flip_flags = np.zeros(len(bounds) - 1, dtype=bool)
            self.pairs += list(zip(keys, bounds[:-1], bounds[1:], flip_flags))

            if augment:
                flip_flags_aug = ~flip_flags
                self.pairs += list(zip(keys, bounds[:-1], bounds[1:], flip_flags_aug))

    # ----------------- 工具函数 -----------------
    def _pad_sequence(self, seq, start, end):
        low = max(start, 0)
        high = min(end, seq.shape[0])
        pad_left = low - start
        pad_right = end - high
        chunk = seq[low:high]
        if pad_left > 0 or pad_right > 0:
            chunk = np.pad(
                chunk,
                ((pad_left, pad_right), (0, 0), (0, 0)),
                mode='edge'
            )
        return chunk

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def __len__(self):
        return len(self.pairs)

    # ----------------- 主逻辑 -----------------
    def __getitem__(self, index):
        seq_i, start_3d, end_3d, flip = self.pairs[index]
        # seq_i 是一个形状为 (3,) 的 numpy 数组
        subject, seq, cam_index = seq_i
        seq_name = (subject, seq, cam_index)

        if self.dataset_type == 'seq2seq':
            # ===== seq2seq 模式：3D 输出为 [chunk_length]，2D 可以是 tds 扩展的长序列 =====
            mid = (start_3d + end_3d) // 2

            if self.tds == 1:
                start_2d = start_3d
                end_2d = end_3d
            else:
                # tds > 1 时，用 pad * tds 决定 2D 的感受野长度（和 H36M 版本一致）
                pad_eff = self.pad * self.tds
                start_2d = mid - pad_eff
                end_2d = mid + pad_eff

            # ----------- 2D pose -------------
            seq_2d = np.asarray(self.poses_2d[seq_name])
            chunk_2d = self._pad_sequence(seq_2d, start_2d, end_2d)

            if self.frame_stride > 1:
                chunk_2d = chunk_2d[::self.frame_stride]

            if flip:
                # 注意：2D 用 kps_left / kps_right
                chunk_2d[:, :, 0] *= -1
                if self.kps_left is not None and self.kps_right is not None:
                    chunk_2d[:, self.kps_left + self.kps_right] = \
                        chunk_2d[:, self.kps_right + self.kps_left]

            # ----------- 3D pose -------------
            chunk_3d = None
            if self.poses_3d is not None:
                seq_3d = np.asarray(self.poses_3d[seq_name])
                # 和你原始 3DHP 代码保持一致：毫米 -> 米
                seq_3d = seq_3d / 1000.0

                chunk_3d = self._pad_sequence(seq_3d, start_3d, end_3d)
                if self.frame_stride > 1:
                    chunk_3d = chunk_3d[::self.frame_stride]

                if flip:
                    chunk_3d[:, :, 0] *= -1
                    if self.joints_left is not None and self.joints_right is not None:
                        chunk_3d[:, self.joints_left + self.joints_right] = \
                            chunk_3d[:, self.joints_right + self.joints_left]

            # ----------- Camera --------------
            if self.cameras is not None:
                cam = np.asarray(self.cameras[seq_name]).copy()
                if flip:
                    cam[2] *= -1
                    cam[7] *= -1
            else:
                cam = np.zeros_like(chunk_3d) if chunk_3d is not None else None

            # 3DHP 原版只返回 (cam, 3D, 2D)
            return cam, chunk_3d, chunk_2d, 0

        else:
            # ===== seq2frame 模式：H36M 风格，2D 有 pad/causal_shift，3D 对应输出帧 =====
            # -------------------- 2D INPUT --------------------
            seq_2d = np.asarray(self.poses_2d[seq_name])

            start_2d = start_3d - self.pad - self.causal_shift
            end_2d = end_3d + self.pad - self.causal_shift

            low_2d = max(start_2d, 0)
            high_2d = min(end_2d, seq_2d.shape[0])
            pad_left = low_2d - start_2d
            pad_right = end_2d - high_2d

            if pad_left != 0 or pad_right != 0:
                chunk_2d = np.pad(
                    seq_2d[low_2d:high_2d],
                    ((pad_left, pad_right), (0, 0), (0, 0)),
                    mode="edge"
                )
            else:
                chunk_2d = seq_2d[low_2d:high_2d]

            if flip:
                chunk_2d[:, :, 0] *= -1
                if self.kps_left is not None and self.kps_right is not None:
                    chunk_2d[:, self.kps_left + self.kps_right] = \
                        chunk_2d[:, self.kps_right + self.kps_left]

            # -------------------- 3D SUPERVISION --------------------
            chunk_3d = None
            if self.poses_3d is not None:
                seq_3d = np.asarray(self.poses_3d[seq_name])
                seq_3d = seq_3d / 1000.0  # 保持和你原 3DHP 一致

                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left = low_3d - start_3d
                pad_right = end_3d - high_3d

                if pad_left != 0 or pad_right != 0:
                    chunk_3d = np.pad(
                        seq_3d[low_3d:high_3d],
                        ((pad_left, pad_right), (0, 0), (0, 0)),
                        mode="edge"
                    )
                else:
                    chunk_3d = seq_3d[low_3d:high_3d]

                if flip:
                    chunk_3d[:, :, 0] *= -1
                    if self.joints_left is not None and self.joints_right is not None:
                        chunk_3d[:, self.joints_left + self.joints_right] = \
                            chunk_3d[:, self.joints_right + self.joints_left]

            # -------------------- Camera --------------------
            if self.cameras is not None:
                cam = np.asarray(self.cameras[seq_name]).copy()
                if flip:
                    cam[2] *= -1
                    cam[7] *= -1
            else:
                cam = np.zeros_like(chunk_3d) if chunk_3d is not None else None

            return cam, chunk_3d, chunk_2d, None


    def _pad_sequence(self, seq, start, end):
        low = max(start, 0)
        high = min(end, seq.shape[0])
        pad_left = low - start
        pad_right = end - high
        chunk = seq[low:high]
        if pad_left > 0 or pad_right > 0:
            chunk = np.pad(chunk, ((pad_left, pad_right), (0, 0), (0, 0)), mode='edge')
        return chunk

class PoseUnchunkedDataset_3DHP(Dataset):
    def __init__(self, poses_2d, poses_3d=None, cameras=None, action_valid=None,
                 pad=0, causal_shift=0, augment=False,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None):

        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.poses_2d = poses_2d
        self.poses_3d = poses_3d
        self.cameras = cameras
        self.pad = pad
        self.causal_shift = causal_shift
        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.keys = list(self.poses_2d.keys())
        cam_1 = torch.tensor([7.32506, 7.32506, -0.0322884, 0.0929296, 0, 0, 0, 0, 0])
        cam_data_1 = [2048, 2048, 10, 10] #width, height, sensorSize_x, sensorSize_y
        cam_2 = torch.tensor([8.770747185, 8.770747185, -0.104908645, 0.104899704, 0, 0, 0, 0, 0])
        cam_data_2 = [1920, 1080, 10, 5.625]  # width, height, sensorSize_x, sensorSize_y
        self.cam_1 = cam_mm_to_pix(cam_1, cam_data_1)
        self.cam_2 = cam_mm_to_pix(cam_2, cam_data_2)


    def __len__(self):
        return len(self.poses_2d)

    def __getitem__(self, idx):
        seq_name = self.keys[idx]
        seq_2d = self.poses_2d[seq_name]
        chunk_2d = np.expand_dims(seq_2d, axis=0)  # (1, T, J, 2)
        chunk_3d = None
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name]
            seq_3d = seq_3d / 1000
            chunk_3d = np.expand_dims(seq_3d, axis=0)
        # cam = None
        
        if seq_name == "TS5" or seq_name == "TS6":
            cam = self.cam_2.clone()
        else:
            cam = self.cam_1.clone()

        # if self.cameras is not None:
        #     cam = np.expand_dims(self.cameras[seq_name], axis=0)
        # else:
        #     cam = np.zeros_like(chunk_3d)  # 占位，保持一致
        cam = np.expand_dims(cam, axis=0)

        if self.augment:
            if cam is not None:
                cam = np.concatenate((cam, cam), axis=0)
                cam[1, 2] *= -1
                cam[1, 7] *= -1
            if chunk_3d is not None:
                chunk_3d = np.concatenate((chunk_3d, chunk_3d), axis=0)
                chunk_3d[1, :, :, 0] *= -1
                chunk_3d[1, :, self.joints_left + self.joints_right] = chunk_3d[1, :, self.joints_right + self.joints_left]
            chunk_2d = np.concatenate((chunk_2d, chunk_2d), axis=0)
            chunk_2d[1, :, :, 0] *= -1
            chunk_2d[1, :, self.kps_left + self.kps_right] = chunk_2d[1, :, self.kps_right + self.kps_left]

        return cam, chunk_3d, chunk_2d, seq_name, None


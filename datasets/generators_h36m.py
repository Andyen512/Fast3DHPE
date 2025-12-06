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
     

class PoseChunkDataset_H36M(Dataset):
    def __init__(self, poses_2d, poses_3d=None, cameras=None, action=None,
                 chunk_length=1, pad=0, causal_shift=0, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 dataset_type= 'seq2frame', frame_stride=1, tds=1):

        self.poses_2d = poses_2d
        self.poses_3d = poses_3d
        self.cameras = cameras
        self.action = action
        self.chunk_length = chunk_length
        self.pad = pad 
        self.causal_shift = causal_shift
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.random = np.random.RandomState(random_seed)
        self.dataset_type = dataset_type
        self.frame_stride = frame_stride
        self.tds = tds

        self.pairs = []  # (seq_idx, start, end, flip)

        for i in range(len(poses_2d)):
            n_frames = poses_2d[i].shape[0]
            n_chunks = (n_frames + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - n_frames) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset

            flip_flags = np.zeros(len(bounds) - 1, dtype=bool)
            self.pairs += list(zip(np.full(len(bounds) - 1, i), bounds[:-1], bounds[1:], flip_flags))

            if augment:
                flip_flags = ~flip_flags
                self.pairs += list(zip(np.full(len(bounds) - 1, i), bounds[:-1], bounds[1:], flip_flags))

    def _pad_sequence(self, seq, start, end):
        low = max(start, 0)
        high = min(end, seq.shape[0])
        pad_left = low - start
        pad_right = end - high
        chunk = seq[low:high]
        if pad_left > 0 or pad_right > 0:
            chunk = np.pad(chunk, ((pad_left, pad_right), (0, 0), (0, 0)), mode='edge')
        return chunk
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        seq_i, start_3d, end_3d, flip = self.pairs[index]
        if self.dataset_type == 'seq2seq':
            mid = (start_3d + end_3d) // 2
            if self.tds == 1:
                start_2d = start_3d
                end_2d   = end_3d
            else:
                pad_eff = self.pad * self.tds
                start_2d = mid - pad_eff
                end_2d   = mid + pad_eff

            # ----------- 2D pose -------------
            seq_2d = np.asarray(self.poses_2d[seq_i])  # Ensure ndarray
            chunk_2d = self._pad_sequence(seq_2d, start_2d, end_2d)

            if self.frame_stride > 1:
                chunk_2d = chunk_2d[::self.frame_stride]

            if flip:
                chunk_2d[:, :, 0] *= -1
                chunk_2d[:, self.kps_left + self.kps_right] = chunk_2d[:, self.kps_right + self.kps_left]

            # ----------- 3D pose -------------
            chunk_3d = None
            if self.poses_3d is not None:
                seq_3d = np.asarray(self.poses_3d[seq_i])
                chunk_3d = self._pad_sequence(seq_3d, start_3d, end_3d)
                if self.frame_stride > 1:
                    chunk_3d = chunk_3d[::self.frame_stride]
                    
                if flip:
                    chunk_3d[:, :, 0] *= -1
                    chunk_3d[:, self.joints_left + self.joints_right] = chunk_3d[:, self.joints_right + self.joints_left]

            # ----------- Camera --------------
            cam = None
            if self.cameras is not None:
                cam = np.asarray(self.cameras[seq_i]).copy()
                if flip:
                    cam[2] *= -1
                    cam[7] *= -1
            else:
                cam = np.zeros_like(chunk_3d)  # 占位，保持一致

            action = self.action[seq_i]  # Ensure ndarray
            return cam, chunk_3d, chunk_2d, action
        else:
            # -------------------- 2D INPUT --------------------
            seq_2d = self.poses_2d[seq_i]
            start_2d = start_3d - self.pad - self.causal_shift
            end_2d = end_3d + self.pad - self.causal_shift

            low_2d = max(start_2d, 0)
            high_2d = min(end_2d, seq_2d.shape[0])
            pad_left = low_2d - start_2d
            pad_right = end_2d - high_2d

            if pad_left != 0 or pad_right != 0:
                chunk_2d = np.pad(seq_2d[low_2d:high_2d],
                                ((pad_left, pad_right), (0, 0), (0, 0)), 
                                mode="edge")
            else:
                chunk_2d = seq_2d[low_2d:high_2d]

            if flip:
                chunk_2d[:, :, 0] *= -1
                chunk_2d[:, self.joints_left + self.joints_right] = chunk_2d[:, self.joints_right + self.joints_left]

            # -------------------- 3D SUPERVISION --------------------
            chunk_3d = None
            if self.poses_3d is not None:
                seq_3d = self.poses_3d[seq_i]

                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left = low_3d - start_3d
                pad_right = end_3d - high_3d

                if pad_left != 0 or pad_right != 0:
                    chunk_3d = np.pad(seq_3d[low_3d:high_3d],
                                    ((pad_left, pad_right), (0, 0), (0, 0)),
                                    mode="edge")
                else:
                    chunk_3d = seq_3d[low_3d:high_3d]

                if flip:
                    chunk_3d[:, :, 0] *= -1
                    chunk_3d[:, self.joints_left + self.joints_right] = chunk_3d[:, self.joints_right + self.joints_left]

            # -------------------- Camera --------------------
            cam = None
            if self.cameras is not None:
                cam = self.cameras[seq_i].copy()
                if flip:
                    cam[2] *= -1
                    cam[7] *= -1

            return cam, chunk_3d, chunk_2d, self.action[seq_i]
     




class PoseUnchunkedDataset_H36M(Dataset):
    def __init__(self, poses_2d, poses_3d=None, cameras=None, action=None,
                 pad=0, causal_shift=0, augment=False,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 dataset_type= 'seq2frame'):

        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.poses_2d = poses_2d
        self.poses_3d = poses_3d
        self.cameras = cameras
        self.action = action
        self.pad = pad
        self.causal_shift = causal_shift
        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def __len__(self):
        return len(self.poses_2d)

    def __getitem__(self, idx):
        seq_2d = self.poses_2d[idx]
        batch_act = self.action[idx]
        chunk_2d = np.expand_dims(seq_2d, axis=0)  # (1, T, J, 2)

        chunk_3d = None
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[idx]
            chunk_3d = np.expand_dims(seq_3d, axis=0)
        
        cam = None
        if self.cameras is not None:
            cam = np.expand_dims(self.cameras[idx], axis=0)
        else:
            cam = np.zeros_like(chunk_3d)  # 占位，保持一致
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

        return cam, chunk_3d, chunk_2d, None, batch_act

# class PoseUnchunkedDataset_H36M(Dataset):
#     def __init__(self, poses_2d, poses_3d=None, cameras=None, action=None,
#                  pad=0, causal_shift=0, augment=False,
#                  kps_left=None, kps_right=None, joints_left=None, joints_right=None,
#                  dataset_type='seq2frame'):

#         assert poses_3d is None or len(poses_3d) == len(poses_2d)
#         assert cameras is None or len(cameras) == len(poses_2d)

#         self.poses_2d = poses_2d
#         self.poses_3d = poses_3d
#         self.cameras  = cameras
#         self.action   = action

#         self.pad          = pad
#         self.causal_shift = causal_shift
#         self.augment      = augment

#         self.kps_left   = kps_left
#         self.kps_right  = kps_right
#         self.joints_left  = joints_left
#         self.joints_right = joints_right

#         self.dataset_type = dataset_type

#     def __len__(self):
#         return len(self.poses_2d)

#     def __getitem__(self, idx):
#         seq_2d = self.poses_2d[idx]      # (T, J, 2)
#         seq_3d = None if self.poses_3d is None else self.poses_3d[idx]   # (T, J, 3) or None
#         cam    = None if self.cameras  is None else self.cameras[idx]    # (C,)

#         # ===================== seq2seq 模式 =====================
#         if self.dataset_type == 'seq2seq':
#             # 2D / 3D 都不 pad，直接整段
#             chunk_2d = np.expand_dims(seq_2d, axis=0)  # (1, T, J, 2)
#             chunk_3d = None
#             if seq_3d is not None:
#                 chunk_3d = np.expand_dims(seq_3d, axis=0)  # (1, T, J, 3)

#             if cam is not None:
#                 cam = np.expand_dims(cam, axis=0)          # (1, C)

#         # ===================== seq2frame 模式 =====================
#         else:
#             # 和 UnchunkedGenerator 一样：2D 两端 pad，3D 原长度
#             # 2D: pad in time dimension
#             start_pad = self.pad + self.causal_shift
#             end_pad   = self.pad - self.causal_shift
#             padded_2d = np.pad(
#                 seq_2d,
#                 ((start_pad, end_pad), (0, 0), (0, 0)),
#                 mode='edge'
#             )  # (T + 2*pad, J, 2)
#             chunk_2d = np.expand_dims(padded_2d, axis=0)    # (1, T+2*pad, J, 2)

#             # 3D: no temporal padding
#             chunk_3d = None
#             if seq_3d is not None:
#                 chunk_3d = np.expand_dims(seq_3d, axis=0)   # (1, T, J, 3)

#             if cam is not None:
#                 cam = np.expand_dims(cam, axis=0)           # (1, C)

#         # ===================== Flip Augmentation =====================
#         if self.augment:
#             # cameras
#             if cam is not None:
#                 cam = np.concatenate((cam, cam), axis=0)    # (2, C)
#                 cam[1, 2] *= -1
#                 cam[1, 7] *= -1

#             # 3D
#             if chunk_3d is not None:
#                 chunk_3d = np.concatenate((chunk_3d, chunk_3d), axis=0)  # (2, T, J, 3)
#                 chunk_3d[1, :, :, 0] *= -1
#                 chunk_3d[1, :, self.joints_left + self.joints_right] = \
#                     chunk_3d[1, :, self.joints_right + self.joints_left]

#             # 2D
#             chunk_2d = np.concatenate((chunk_2d, chunk_2d), axis=0)      # (2, T or T+2*pad, J, 2)
#             chunk_2d[1, :, :, 0] *= -1
#             chunk_2d[1, :, self.kps_left + self.kps_right] = \
#                 chunk_2d[1, :, self.kps_right + self.kps_left]

#         act = None if self.action is None else self.action[idx]
#         return cam, chunk_3d, chunk_2d, None, act
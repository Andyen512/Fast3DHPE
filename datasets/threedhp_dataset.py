# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from .skeleton import Skeleton
from .mocap_dataset import MocapDataset
from .camera import normalize_screen_coordinates, image_coordinates
import os
import torch
from torch.utils.data import Dataset
from functools import reduce
from remote_pdb import set_trace

class ThreeDHPDataset():
    def __init__(self, root, name, remove_static_joints=True):
        super().__init__()

        train_path = os.path.join(root, f"data_train_{name}_ori.npz")
        test_path = os.path.join(root, f"data_test_{name}_ori.npz")

        # Load serialized dataset
        self.data_train = np.load(train_path, allow_pickle=True)['data'].item()
        self.data_test = np.load(test_path, allow_pickle=True)['data'].item()
        self.joints_left=[5, 6, 7, 11, 12, 13]
        self.joints_right=[2, 3, 4, 8, 9, 10]
        self.kps_left=[5, 6, 7, 11, 12, 13]
        self.kps_right=[2, 3, 4, 8, 9, 10]
        self.skeleton = None
        
        # if remove_static_joints:
        #     # Bring the skeleton to 17 joints instead of the original 32
        #     self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
        #     # Rewire shoulders to the correct parents
        #     self._skeleton._parents[11] = 8
        #     self._skeleton._parents[14] = 8


#####################################
# data loader with two output
#####################################
class PoseBuffer(Dataset):
    def __init__(self, poses_3d, poses_2d, score=None):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        # print('Generating {} poses...'.format(self._poses_3d.shape[0]))
        self.skeleton = None
        self.joints_left=[4, 5, 6, 10, 11, 12]
        self.joints_right=[1, 2, 3, 13, 14, 15]
        self.kps_left=[4, 5, 6, 10, 11, 12]
        self.kps_right=[1, 2, 3, 13, 14, 15]
    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d

    def __len__(self):
        return len(self._poses_2d)

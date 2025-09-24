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
        
        # if remove_static_joints:
        #     # Bring the skeleton to 17 joints instead of the original 32
        #     self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
        #     # Rewire shoulders to the correct parents
        #     self._skeleton._parents[11] = 8
        #     self._skeleton._parents[14] = 8
            

# This code is based on https://github.com/Mathux/ACTOR.git
import numpy as np
import torch

import contextlib

from smplx import create as smplx_create
from smplx.lbs import vertices2joints


# action2motion_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 24, 38]
# change 0 and 8
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]

from utils.config import SMPL_MODEL_PATH, JOINT_REGRESSOR_TRAIN_EXTRA

JOINTSTYPE_ROOT = {"a2m": 0, # action2motion
                   "smpl": 0,
                   "a2mpl": 0, # set(smpl, a2m)
                   "vibe": 8}  # 0 is the 8 position: OP MidHip below

JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]


# adapted from VIBE/SPIN to output smpl_joints, vibe joints and action2motion joints
class SMPL(torch.nn.Module):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, model_path=SMPL_MODEL_PATH, model_type='smpl', **kwargs):
        super(SMPL, self).__init__()
        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            self.model = smplx_create(model_path=model_path, model_type=model_type, **kwargs)
            
        self.num_betas = self.model.num_betas if hasattr(self.model, 'num_betas') else 10
            
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        vibe_indexes = np.array([JOINT_MAP[i] for i in JOINT_NAMES])
        a2m_indexes = vibe_indexes[action2motion_joints]
        
        # Determine number of joints in standard output
        # For SMPL-X, we want more than 24.
        num_joints = self.model.NUM_BODY_JOINTS + 1 + 30 + 3 + 20 # usually 127 for SMPL-X
        # Actually, let's just use all joints available if it's more than 24.
        # But for map initialization, we need a fixed number or a way to handle it.
        # Standard SMPL has 24 joints.
        smpl_indexes = np.arange(127) if model_type == 'smplx' else np.arange(24)
        
        a2mpl_indexes = np.unique(np.r_[np.arange(24), a2m_indexes])

        self.maps = {"vibe": vibe_indexes,
                     "a2m": a2m_indexes,
                     "smpl": smpl_indexes,
                     "a2mpl": a2mpl_indexes}

        
    def forward(self, *args, **kwargs):
        # Check for batch size > 1 and handle it if model is initialized for size 1
        batch_size = 1
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                batch_size = max(batch_size, v.shape[0])
        
        if batch_size > 1:
            # Slower path but handles broadcasting mismatch in smplx library
            all_outputs = []
            for i in range(batch_size):
                single_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                        single_kwargs[k] = v[i:i+1]
                    else:
                        single_kwargs[k] = v
                
                single_output = self.model(*args, **single_kwargs)
                all_outputs.append(single_output)
            
            # Aggregate outputs
            smpl_output = {}
            for key in all_outputs[0].keys():
                vals = [out[key] for out in all_outputs if out[key] is not None]
                if vals:
                    smpl_output[key] = torch.cat(vals, dim=0)
                else:
                    smpl_output[key] = None
        else:
            smpl_output = self.model(*args, **kwargs)
        
        # Normalize access (smpl_output could be a dict or a ModelOutput object)
        if isinstance(smpl_output, dict):
            vertices = smpl_output['vertices']
            joints = smpl_output['joints']
        else:
            vertices = smpl_output.vertices
            joints = smpl_output.joints

        if vertices.shape[1] == self.J_regressor_extra.shape[1]:
            extra_joints = vertices2joints(self.J_regressor_extra, vertices)
            all_joints = torch.cat([joints, extra_joints], dim=1)
        else:
            # For SMPL-X or other models where vertex count doesn't match the SMPL regressor
            all_joints = joints

        output = {"vertices": vertices}

        for joinstype, indexes in self.maps.items():
            # For SMPL-X, smpl_output.joints might have different ordering or more joints.
            # However, for basic vibe/a2m evaluation, we often rely on vertices2joints extra regressor.
            # We keep it as is and see.
            if indexes.max() < all_joints.shape[1]:
                output[joinstype] = all_joints[:, indexes]
            
        return output
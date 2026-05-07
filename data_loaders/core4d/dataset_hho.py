import copy
import glob
import json
import os
from os.path import join, dirname, abspath, isfile, isdir
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), ".."))
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml
from data_loaders.humanml.utils.get_opt import get_opt

try:
    from data.tools import vertex_normals
    from data.utils import markerset_ssm67_smplx
except ImportError:
    print("Warning: Could not import data.tools and data.utils. Some features may not work.")
    
sys.path.insert(0, join(dirname(abspath(__file__)), "../../../.."))
try:
    from data_loaders.core4d.data_processing.smplx import smplx
    from data_loaders.core4d.dataset_statistics.train_test_split import load_train_test_split, load_train_test_split_retargeted
    import open3d as o3d
except ImportError:
    print("Warning: Could not import SMPLX or dataset utilities.")

from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d

MODEL_PATH = "/share/human_model/models"


class CORE4DDataset(Dataset):
    """
    CORE4D Dataset compatible with ShelfAssembly format.
    Extracts motion data and converts to 6D rotation representation.
    """
    def __init__(self, split='train', past_len=15, future_len=15, sample_rate=1,
                 smplx_model_dir="", test_set="", datapath="./dataset/core4d_opt.txt", **kwargs):
        """
        Args:
            split: 'train' or 'test'
            past_len: number of past frames
            future_len: number of future frames
            sample_rate: frame sampling rate
            dataset_root: root path to CORE4D dataset
            smplx_model_dir: path to SMPLX model directory
            test_set: 'all', 'seen', or 'unseen' for test mode
            **kwargs: additional arguments for compatibility
        """
        abs_base_path = kwargs.get("abs_path", ".")
        dataset_opt_path = os.path.join(abs_base_path, datapath)
        device = kwargs.get("device", None)
        self.opt = get_opt(dataset_opt_path, device)
        dataset_root = self.opt.data_root

        self.obj_categories = ["chair", "desk", "board", "box", "bucket", "stick"]
        # select whether adding real/retargeted data
        self.add_real = True
        self.add_retarget = False
        
        # get sequence dirs
        try:
            train_sequence_names, test_sequence_names_seen_obj, test_sequence_names_unseen_obj = load_train_test_split()
        except Exception as e:
            print(f"Warning: Could not load train/test split: {e}")
            train_sequence_names = []
            test_sequence_names_seen_obj = []
            test_sequence_names_unseen_obj = []
            
        if not self.add_real:
            train_sequence_names = []
        if self.add_retarget:
            try:
                train_sequence_names_retargeted = load_train_test_split_retargeted()
                train_sequence_names += train_sequence_names_retargeted
            except Exception as e:
                print(f"Warning: Could not load retargeted data: {e}")
                
        if split == 'train':
            sequence_names = train_sequence_names
        elif split == "test":
            if test_set == "all":
                sequence_names = test_sequence_names_seen_obj + test_sequence_names_unseen_obj
            elif test_set == "seen":
                sequence_names = test_sequence_names_seen_obj
            elif test_set == "unseen":
                sequence_names = test_sequence_names_unseen_obj
            else:
                raise NotImplementedError
        else:
            raise Exception('split must be train or test.')
        self.seq_dirs = []
        for sn in sequence_names:
            seq_dir = join(dataset_root, self.opt.motion_dir, sn.replace(".", "/"))
            if isfile(join(seq_dir, "data.npz")):
                self.seq_dirs.append(seq_dir)
        self.seq_dirs.sort()
        
        print(f"###### Number of CORE4D sequences in total = {len(self.seq_dirs)}")
        
        self.repeat_ratio = 1
        self.past_len = past_len
        self.future_len = future_len
        self.sample_rate = sample_rate
        
        # Load data in shelf_assembly format
        self.motion_data = []  # List of motion clips with 6D rotations
        self.raw_data = []    # Raw data for reference
        self.idx2frame = []   # (seq_id, frame_idx, bias)
        
        seq_idx = 0
        for seq_dir in tqdm(self.seq_dirs, desc="Loading CORE4D sequences"):
            try:
                d = np.load(join(seq_dir, "data.npz"), allow_pickle=True)["data"].item()
                clip_name, seq_name = seq_dir.split("/")[-2:]
                N_frame = d["N_frame"]
                obj_model_path = d["obj_model_path"]
                
                # Filter by object category
                if (obj_model_path.find("vchair") > -1) or (obj_model_path.find("vtable") > -1):
                    obj_cat = obj_model_path.split("/")[-2].split("_")[1][1:6].replace("table", "desk")
                else:
                    obj_cat = obj_model_path.split("/")[-2]
                
                if obj_cat not in self.obj_categories:
                    continue
                
                # Extract motion data in shelf_assembly format
                human_params_p1 = d["human_params"]["person1"]
                human_params_p2 = d["human_params"]["person2"]

                # Process person 1 motion
                motion_p1 = self._extract_motion_features(
                    human_params_p1, seq_idx, clip_name, seq_name, "Person1"
                )
                self.motion_data.append(motion_p1)
                
                # Process person 2 motion
                motion_p2 = self._extract_motion_features(
                    human_params_p2, seq_idx, clip_name, seq_name, "Person2"
                )
                self.motion_data.append(motion_p2)
                
                # Store raw data for later access
                self.raw_data.append(d)
                
                # Create frame indices for motion clips
                fragment = (past_len + future_len) * sample_rate
                for i in range(N_frame // fragment):
                    self.idx2frame.append((len(self.motion_data) - 2, i * fragment, fragment))
                
                seq_idx += 1
            except Exception as e:
                print(f"Error loading sequence {seq_dir}: {e}")
                continue
        
        print(f"###### Loaded {len(self.motion_data)} motion clips from CORE4D")
        assert len(self.idx2frame) > 0, "No valid motion clips found in dataset"
    
    def _extract_motion_features(self, human_params, seq_idx, clip_name, seq_name, person_id):
        """
        Extract motion features from human parameters in shelf_assembly format.
        
        Args:
            human_params: dict with 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl'
            seq_idx: sequence index
            clip_name: clip name string
            seq_name: sequence name string
            person_id: person identifier string
            
        Returns:
            dict with motion features converted to 6D rotation format
        """
        N_frame = len(human_params['transl'])
        
        # Convert global orientation from axis-angle to 6D rotation
        global_orient_aa = torch.from_numpy(human_params['global_orient']).float()
        global_orient_6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_aa))
        
        # Convert body pose from axis-angle to 6D rotation
        body_pose_aa = torch.from_numpy(human_params['body_pose']).float()
        body_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose_aa))
        
        # Convert left hand pose from axis-angle to 6D rotation
        left_hand_pose = human_params['left_hand_pose'].reshape(-1, 3, 4)
        left_hand_rot  = left_hand_pose[:, :, :3]
        left_hand_pose_aa = torch.from_numpy(left_hand_rot).float()
        left_hand_pose_6d = matrix_to_rotation_6d(left_hand_pose_aa)
        
        # Convert right hand pose from axis-angle to 6D rotation
        right_hand_pose = human_params['right_hand_pose'].reshape(-1, 3, 4)
        right_hand_rot  = right_hand_pose[:, :, :3]
        right_hand_pose_aa = torch.from_numpy(right_hand_rot).float()
        right_hand_pose_6d = matrix_to_rotation_6d(right_hand_pose_aa)
        
        # Root position (translation)
        root_pos = torch.from_numpy(human_params['transl']).float()
        
        motion_dict = {
            'global_orient': global_orient_6d,    # (N_frame, 6)
            'body_pose': body_pose_6d,            # (N_frame, 252) = 42 joints * 6
            'left_hand_pose': left_hand_pose_6d,  # (N_frame, 60) = 10 joints * 6
            'right_hand_pose': right_hand_pose_6d,# (N_frame, 60) = 10 joints * 6
            'root_pos': root_pos,                 # (N_frame, 3)
            'clip_name': clip_name,
            'seq_name': seq_name,
            'person_id': person_id,
            'seq_idx': seq_idx,
            'no': seq_idx,  # For compatibility with shelf_assembly
        }
        
        return motion_dict
    
    def __len__(self):
        return len(self.idx2frame) * self.repeat_ratio
    
    def __getitem__(self, idx):
        """
        Get a motion clip in shelf_assembly format.
        
        Returns:
            motion: dict with motion features (global_orient, body_pose, etc.) in 6D rotation format
            annotation: dict with metadata
        """
        idx = idx % len(self.idx2frame)
        seq_id, frame_idx, bias = self.idx2frame[idx]
        
        # Get motion data
        motion = copy.deepcopy(self.motion_data[seq_id])
        
        # Determine start frame with some randomness
        if isinstance(bias, (list, np.ndarray)):
            start_offset = np.random.choice(bias) if len(bias) > 0 else 0
        else:
            start_offset = np.random.randint(0, max(1, bias))
        
        start_frame = frame_idx + start_offset
        end_frame = start_frame + (self.past_len + self.future_len) * self.sample_rate
        
        # Extract clip
        motion_clip = {}
        for key in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'root_pos']:
            motion_clip[key] = motion[key][start_frame:end_frame:self.sample_rate]
        
        # Keep metadata
        for key in ['clip_name', 'seq_name', 'person_id', 'seq_idx', 'no']:
            if key in motion:
                motion_clip[key] = motion[key]
        
        # Create annotation
        annotation = {
            'no': motion.get('no', 0),
            'clip_name': motion.get('clip_name', ''),
            'seq_name': motion.get('seq_name', ''),
            'person_id': motion.get('person_id', ''),
            'valid_length': min(self.past_len + self.future_len, end_frame - start_frame),
            'dataset': 'CORE4D'
        }
        
        return motion_clip, annotation


if __name__ == "__main__":
    """
    Example usage of CORE4D dataset in shelf_assembly compatible format.
    
    The dataset outputs motion features with the following format:
    - global_orient: (N_frames, 6) - 6D rotation representation
    - body_pose: (N_frames, 252) - 42 joints × 6D rotation
    - left_hand_pose: (N_frames, 60) - 10 joints × 6D rotation
    - right_hand_pose: (N_frames, 60) - 10 joints × 6D rotation
    - root_pos: (N_frames, 3) - root translation
    """
    
    # Initialize dataset
    dataset = CORE4DDataset(
        split='train',
        past_len=15,
        future_len=15,
        sample_rate=1,
        dataset_root="/path/to/core4d/dataset",
        smplx_model_dir="/path/to/smplx/models",
        test_set="all"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    motion, annotation = dataset[0]
    
    print("\n=== Motion features (ShelfAssembly compatible format) ===")
    print(f"global_orient shape: {motion['global_orient'].shape}")  # (N_frames, 6)
    print(f"body_pose shape: {motion['body_pose'].shape}")  # (N_frames, 252)
    print(f"left_hand_pose shape: {motion['left_hand_pose'].shape}")  # (N_frames, 60)
    print(f"right_hand_pose shape: {motion['right_hand_pose'].shape}")  # (N_frames, 60)
    print(f"root_pos shape: {motion['root_pos'].shape}")  # (N_frames, 3)
    
    print("\n=== Annotation ===")
    for key, value in annotation.items():
        print(f"{key}: {value}")


"""
CoMaD Dataset - Two-Person Interaction Motion Dataset

This module provides both the original CoMaD dataset class and a ShelfAssembly-compatible version.

Classes:
    - CoMaD: Original implementation (kept for backward compatibility)
    - CoMaDDataset: ShelfAssembly-compatible implementation (recommended)
"""

import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from tqdm import tqdm

try:
    from interact.utils.read_json_data import read_json, get_pose_history, missing_data
    from hydra import compose
except ImportError:
    print("Warning: Could not import interact utilities or hydra. Some features may not work.")
    read_json = None
    compose = None
    get_pose_history = None
    missing_data = None


def convert_time_to_frame(time, hz, offset):
    """Convert time string (MM:SS) to frame index."""
    mins = int(time[:time.find(':')])
    secs = int(time[time.find(':')+1:])
    return ((mins*60 + secs) * hz) + offset


class CoMaD(Dataset):
    """
    Original CoMaD Dataset implementation (kept for backward compatibility).

    Returns separate input and output tensors for Alice and Bob.
    For new code, use CoMaDDataset which has ShelfAssembly-compatible format.
    """

    def __init__(
            self,
            input_n=15,
            output_n=15,
            sample_rate=120,
            output_rate=15,
            split='train',
    ):
        cfg = compose(config_name="datasets", overrides=[])
        self.data_dir = cfg.comad
        self.input_frames = input_n
        self.output_frames = output_n
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.mapping_json = cfg.comad_mapping

        self.alice_input, self.alice_output = [], []
        self.bob_input, self.bob_output = [], []
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n

        joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut',
                        'LHandOut', 'RHandOut']

        mapping = read_json(self.mapping_json)
        self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        self.add_comad_dataset()

    def add_comad_dataset(self):
        for task in os.listdir(f'{self.data_dir}/{self.split}'):
            for episode in os.listdir(f'{self.data_dir}/{self.split}/{task}/HH'):
                print(f'Episode: {self.data_dir}/{self.split}/{task}/HH/{episode}')
                try:
                    json_data = read_json(f'{self.data_dir}/{self.split}/{task}/HH/{episode}/data.json')
                except:
                    continue
                metadata = read_json(f'{self.data_dir}/{self.split}/{task}/HH/{episode}/metadata.json')
                alice_name, bob_name = metadata.keys()
                downsample_rate = self.sample_rate // self.output_rate
                alice_tensor = get_pose_history(json_data,
                                alice_name)[::downsample_rate, self.joint_used]
                bob_tensor = get_pose_history(json_data,
                                bob_name)[::downsample_rate, self.joint_used]
                # chop the tensor into a bunch of slices of size sequence_len
                for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                    end_frame = start_frame + self.sequence_len
                    if missing_data(alice_tensor[start_frame:end_frame]) or \
                        missing_data(bob_tensor[start_frame:end_frame]):
                        # print("MISSING DATA")
                        continue
                    self.alice_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                    self.alice_output.append(alice_tensor[start_frame+self.input_n:end_frame])

                    self.bob_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                    self.bob_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                    ### Flip Alice and Bob
                    self.alice_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                    self.alice_output.append(bob_tensor[start_frame+self.input_n:end_frame])

                    self.bob_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                    self.bob_output.append(alice_tensor[start_frame+self.input_n:end_frame])
                # break
        print(len(self.alice_input))

    def __len__(self):
        return len(self.alice_input)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 9 joints, 3d)
        return self.alice_input[idx], self.alice_output[idx], self.bob_input[idx], self.bob_output[idx]


class CoMaDDataset(Dataset):
    """
    CoMaD Dataset compatible with ShelfAssembly format.

    Extracts multi-person interaction motion data and converts to ShelfAssembly-compatible format.

    The dataset handles two-person interaction (Alice and Bob) and outputs motion in a
    format compatible with ShelfAssembly-based motion generation pipelines.

    Output format:
    - motion: dict with joint positions and metadata
    - annotation: dict with task, episode, and other metadata

    This is the recommended implementation for use with ShelfAssembly-based training.
    """

    def __init__(
            self,
            input_n=15,
            output_n=15,
            sample_rate=120,
            output_rate=15,
            split='train',
            data_dir=None,
            mapping_json=None,
            **kwargs
    ):
        """
        Args:
            input_n (int): Number of input frames (past context)
            output_n (int): Number of output frames (future prediction target)
            sample_rate (int): Original sampling rate in Hz (default: 120)
            output_rate (int): Output sampling rate in Hz (default: 15)
            split (str): Dataset split - 'train', 'test', or 'val'
            data_dir (str): Root directory of CoMaD dataset
            mapping_json (str): Path to joint mapping JSON file
            **kwargs: Additional arguments for compatibility
        """
        self.input_frames = input_n
        self.output_frames = output_n
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n

        # Handle configuration loading
        if data_dir is None or mapping_json is None:
            try:
                cfg = compose(config_name="datasets", overrides=[])
                self.data_dir = data_dir or cfg.comad
                self.mapping_json = mapping_json or cfg.comad_mapping
            except Exception as e:
                print(f"Warning: Could not load config with compose: {e}")
                print("Please provide data_dir and mapping_json explicitly")
                self.data_dir = data_dir
                self.mapping_json = mapping_json
        else:
            self.data_dir = data_dir
            self.mapping_json = mapping_json

        # Joint names to extract (9 joints for CoMaD)
        joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                       'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut',
                       'LHandOut', 'RHandOut']

        # Load joint mapping
        try:
            mapping = read_json(self.mapping_json)
            self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        except Exception as e:
            print(f"Warning: Could not load joint mapping: {e}")
            # Fallback to default indices
            self.joint_used = np.arange(9)

        # Data storage
        self.motion_clips = []  # List of motion clip dictionaries
        self.metadata_list = []  # List of annotation dictionaries

        # Load dataset
        self._load_comad_data()

    def _load_comad_data(self):
        """
        Load all CoMaD sequences and create motion clips.

        For each valid sequence segment, creates two clips:
        1. Alice as primary person
        2. Bob as primary person (flipped)

        This doubles the dataset size for better diversity.
        """
        if self.data_dir is None:
            print("Error: data_dir is not set")
            return

        dataset_path = os.path.join(self.data_dir, self.split)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return

        downsample_rate = self.sample_rate // self.output_rate

        print(f"Loading CoMaD {self.split} dataset from {dataset_path}")

        # Iterate through task directories
        for task in sorted(os.listdir(dataset_path)):
            task_path = os.path.join(dataset_path, task, 'HH')
            if not os.path.isdir(task_path):
                continue

            # Iterate through episodes
            for episode in sorted(os.listdir(task_path)):
                episode_path = os.path.join(task_path, episode)

                try:
                    # Read data and metadata
                    json_data = read_json(os.path.join(episode_path, "data.json"))
                    metadata = read_json(os.path.join(episode_path, "metadata.json"))
                except Exception as e:
                    print(f"Error loading {episode_path}: {e}")
                    continue

                try:
                    # Get person names
                    alice_name, bob_name = list(metadata.keys())

                    # Extract motion tensors with downsampling and joint selection
                    alice_tensor = get_pose_history(
                        json_data, alice_name
                    )[::downsample_rate, self.joint_used]
                    bob_tensor = get_pose_history(
                        json_data, bob_name
                    )[::downsample_rate, self.joint_used]

                    # Create motion clips by sliding window
                    clip_idx = 0
                    for start_frame in range(alice_tensor.shape[0] - self.sequence_len):
                        end_frame = start_frame + self.sequence_len

                        # Check for missing data
                        if (missing_data(alice_tensor[start_frame:end_frame]) or
                                missing_data(bob_tensor[start_frame:end_frame])):
                            continue

                        # Extract input and output portions
                        alice_full = alice_tensor[start_frame:end_frame]
                        bob_full = bob_tensor[start_frame:end_frame]

                        # Clip 1: Alice as primary
                        clip_1 = {
                            'alice_motion': torch.from_numpy(alice_full).float(),
                            'bob_motion': torch.from_numpy(bob_full).float(),
                            'task': task,
                            'episode': episode,
                            'clip_idx': clip_idx,
                            'person_id': 'Alice'
                        }

                        self.motion_clips.append(clip_1)
                        self.metadata_list.append({
                            'no': len(self.motion_clips) - 1,
                            'task': task,
                            'episode': episode,
                            'clip_idx': clip_idx,
                            'person_id': 'Alice',
                            'valid_length': self.sequence_len,
                            'dataset': 'CoMaD',
                            'verb': task
                        })

                        # Clip 2: Bob as primary (flipped version)
                        clip_2 = {
                            'alice_motion': torch.from_numpy(bob_full).float(),
                            'bob_motion': torch.from_numpy(alice_full).float(),
                            'task': task,
                            'episode': episode,
                            'clip_idx': clip_idx,
                            'person_id': 'Bob'
                        }

                        self.motion_clips.append(clip_2)
                        self.metadata_list.append({
                            'no': len(self.motion_clips) - 1,
                            'task': task,
                            'episode': episode,
                            'clip_idx': clip_idx,
                            'person_id': 'Bob',
                            'valid_length': self.sequence_len,
                            'dataset': 'CoMaD',
                            'verb': task
                        })

                        clip_idx += 1

                except Exception as e:
                    print(f"Error processing episode {episode}: {e}")
                    continue

        print(f"Loaded {len(self.motion_clips)} motion clips from CoMaD ({self.split})")

    def __len__(self):
        """Return the number of motion clips in the dataset."""
        return len(self.motion_clips)

    def __getitem__(self, idx):
        """
        Get a motion sample in ShelfAssembly-compatible format.

        Args:
            idx (int): Index of the motion clip

        Returns:
            motion (dict): Motion features with keys:
                - 'alice_joints': (sequence_len, 9, 3) - Alice's joint positions
                - 'bob_joints': (sequence_len, 9, 3) - Bob's joint positions
                - 'pose': (sequence_len, 27) - Flattened Alice pose for model input
                - 'other_pose': (sequence_len, 27) - Flattened Bob pose
                - 'task': str - Task name
                - 'episode': str - Episode identifier
                - 'person_id': str - Primary person (Alice or Bob)

            annotation (dict): Metadata with keys:
                - 'no': int - Clip index
                - 'task': str - Task name
                - 'episode': str - Episode identifier
                - 'clip_idx': int - Clip index within episode
                - 'person_id': str - Primary person identifier
                - 'valid_length': int - Number of valid frames
                - 'dataset': str - Dataset name ('CoMaD')
        """
        clip = self.motion_clips[idx]
        metadata = self.metadata_list[idx]

        # Create motion dictionary
        # CoMaD has 9 joints with 3D coordinates
        # alice_motion shape: (sequence_len, 9, 3)

        motion_dict = {
            # Raw joint positions
            'alice_joints': clip['alice_motion'],      # (sequence_len, 9, 3)
            'bob_joints': clip['bob_motion'],          # (sequence_len, 9, 3)
            # Flattened version for model input
            'pose': clip['alice_motion'].reshape(
                clip['alice_motion'].shape[0], -1
            ),                                          # (sequence_len, 27)
            'other_pose': clip['bob_motion'].reshape(
                clip['bob_motion'].shape[0], -1
            ),                                          # (sequence_len, 27)
            # Metadata
            'task': clip['task'],
            'episode': clip['episode'],
            'person_id': clip['person_id'],
        }

        annotation_dict = copy.deepcopy(metadata)

        return motion_dict, annotation_dict


if __name__ == "__main__":
    """
    Example usage of CoMaDDataset in ShelfAssembly-compatible format.
    """
    # Initialize dataset
    dataset = CoMaDDataset(
        input_n=15,
        output_n=15,
        split='train',
        data_dir='/path/to/comad/dataset',
        mapping_json='/path/to/joint_mapping.json'
    )

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    try:
        motion, annotation = dataset[0]

        print("\n=== Motion features (ShelfAssembly compatible format) ===")
        print(f"alice_joints shape: {motion['alice_joints'].shape}")  # (30, 9, 3)
        print(f"bob_joints shape: {motion['bob_joints'].shape}")      # (30, 9, 3)
        print(f"pose shape: {motion['pose'].shape}")                  # (30, 27)
        print(f"other_pose shape: {motion['other_pose'].shape}")      # (30, 27)

        print("\n=== Annotation ===")
        for key, value in annotation.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error getting sample: {e}")
        print("Make sure to provide valid data_dir and mapping_json paths")

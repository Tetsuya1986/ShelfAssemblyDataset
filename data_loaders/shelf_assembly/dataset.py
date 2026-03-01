import copy
import glob
import json
import os

import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from tqdm.contrib import tzip

from data_loaders.humanml.utils.get_opt import get_opt
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d


class ShelfAssemblyDataset(data.Dataset):
    def __init__(
        self, mode, datapath="./dataset/shelf_assembly_opt.txt", split="train", task="generation", **kwargs
    ):
        self.task = task
        abs_base_path = kwargs.get("abs_path", ".")
        dataset_opt_path = os.path.join(abs_base_path, datapath)
        device = kwargs.get("device", None)
        self.opt = get_opt(dataset_opt_path, device)

        self.is_autoregressive = kwargs.get('autoregressive', False)
        self.split = split

        # Load split IDs
        self.split_ids = self.load_split_ids(os.path.join(abs_base_path, f"dataset/shelf_assembly_{self.split}.txt"))

        # Save prediction-specific parameters directly to self, not opt
        if self.task == "prediction":
            self.input_seconds = kwargs["input_seconds"]
            self.prediction_seconds = kwargs["prediction_seconds"]
            self.stride = kwargs["stride"]

        self.use_envcam  = kwargs.get('use_envcam')
        self.use_headcam = kwargs.get('use_headcam')

        print(f"Loading dataset {self.opt.dataset_name} (split: {self.split}) ...")
        motion_data = self.load_motion_data(self.opt.motion_dir, mode, device)
        annotation = self.load_annotation(self.opt.text_dir, mode, device)

        # Lazy loading setup (using pre-encoded CLIP features)
        self.headcam_source = {}
        if self.use_headcam:
            self.headcam_source = self.get_cam_source_mapping(self.opt.headcam_clip_dir)
            if not self.headcam_source:
                print(f"Warning: No headcam CLIP features found in {self.opt.headcam_clip_dir}")

        self.envcam_source = {}
        if self.use_envcam:
            self.envcam_source = self.get_cam_source_mapping(self.opt.envcam_clip_dir)
            if not self.envcam_source:
                print(f"Warning: No envcam CLIP features found in {self.opt.envcam_clip_dir}")

        self.motion_clip, self.annotation_clip = self.extract_motion_clip(
            mode, motion_data, annotation,
            self.opt.fps, self.opt.envcam_fps, self.opt.headcam_fps,
            self.opt.max_motion_length, device)
        
        # For joint_motion_prediction task, reorganize data to pair Main and Sub motions
        if self.task == 'joint_motion_prediction':
            self.motion_clip, self.annotation_clip = self._organize_joint_motion_pairs(
                self.motion_clip, self.annotation_clip)
        
        self.pre_load = kwargs.get('pre_load_features', False)
        self.feature_cache = {} if self.pre_load else None
        self.dir_cache = {} # Caches list of files in each camera directory
        if self.pre_load and (self.use_headcam or self.use_envcam):
            self.pre_load_features()

    def load_split_ids(self, split_file_path):
        split_ids = set()
        with open(split_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    split_ids.add(int(line))
        return split_ids

    def load_motion_data(self, motion_dir, mode, device):
        # output list[dic]. dic={'no','main_sub','global_orient','root_pos','body_ppose','right_hand_pose','left_hand_pose'}
        data_list = []
        file_pattern = os.path.join(motion_dir, "*.npz")
        files = sorted(glob.glob(file_pattern))
        
        for filepath in tqdm(files, desc="Loading motion data"):
            filename = os.path.basename(filepath)
            if mode in ["action", "action_task", "action_taskcommon", "action_task_taskcommon"]:
                # Check filename pattern if necessary (e.g. HH check)
                if len(filename) >= 19 and filename[17:19] == "HH":
                    no = int(filename[:6])
                        
                    # Filter by split
                    if no not in self.split_ids:
                        continue

                    main_sub = filename[22:26].replace("_", "")
                    dic = {}
                    dic["no"] = no
                    dic["main_sub"] = main_sub

                    try:
                        data = np.load(filepath)
                        
                        # global_orient
                        tensor = torch.tensor(data["global_orient"])
                        dic["global_orient"] = matrix_to_rotation_6d(
                            axis_angle_to_matrix(tensor)
                        )
                        
                        # root_pos
                        dic["root_pos"] = torch.tensor(data["root_pos"])
                        
                        # body_pose
                        tensor = torch.tensor(data["body_pose"])
                        dic["body_pose"] = matrix_to_rotation_6d(
                            axis_angle_to_matrix(tensor)
                        )
                        
                        # right_hand_pose
                        tensor = torch.tensor(data["right_hand_pose"])
                        dic["right_hand_pose"] = matrix_to_rotation_6d(
                            axis_angle_to_matrix(tensor)
                        )
                        
                        # left_hand_pose
                        tensor = torch.tensor(data["left_hand_pose"])
                        dic["left_hand_pose"] = matrix_to_rotation_6d(
                            axis_angle_to_matrix(tensor)
                        )
                        
                        data_list.append(dic)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue

        return data_list

    def load_annotation(self, text_dir, mode, device):
        # output list[dic]. dic={'no','main_sub','shelf_id','ass_dis','data'}
        data_list = []
        if mode in ["action", "action_task", "action_taskcommon", "action_task_taskcommon"]:
            for root, _, files in os.walk(text_dir):
                sorted_files = sorted(files)
                for filename in tqdm(sorted_files, desc="Loading annnotation data"):
                    if (
                        filename.lower().endswith("_action.json")
                        and len(filename) >= 19
                        and filename[17:19] == "HH"
                    ):
                        no = int(filename[:6])

                        # Filter by split
                        if no not in self.split_ids:
                            continue

                        filepath = os.path.join(root, filename)
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            dic = {}
                            dic["no"] = data["no"]
                            dic["shelf_id"] = data["shelf_id"]
                            dic["ass_dis"] = data["ass_dis"]
                            dic["main_sub"] = "Main"
                            dic["data"] = data["Main_data"]
                            data_list.append(copy.deepcopy(dic))
                            dic["main_sub"] = "Sub"
                            dic["data"] = data["Sub_data"]
                            data_list.append(copy.deepcopy(dic))

        if mode in ["action_task", "action_task_taskcommon"]:
            for root, _, files in os.walk(text_dir):
                sorted_files = sorted(files)
                for filename in tqdm(sorted_files, desc="Loading annnotation data"):
                    if (
                        filename.lower().endswith("_task_specific.json")
                        and len(filename) >= 19
                        and filename[17:19] == "HH"
                    ):
                        no = int(filename[:6])

                        # Filter by split
                        if no not in self.split_ids:
                            continue

                        filepath = os.path.join(root, filename)
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            results = [item for item in data_list if item.get("no") == data["no"]]
                            for res in results:
                                res['data_task'] = data['data']

        if mode in ["action_taskcommon", "action_task_taskcommon"]:
            for root, _, files in os.walk(text_dir):
                sorted_files = sorted(files)
                for filename in tqdm(sorted_files, desc="Loading annnotation data"):
                    if (
                        filename.lower().endswith("_task_common.json")
                        and len(filename) >= 19
                        and filename[17:19] == "HH"
                    ):
                        no = int(filename[:6])

                        # Filter by split
                        if no not in self.split_ids:
                            continue

                        filepath = os.path.join(root, filename)
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            results = [item for item in data_list if item.get("no") == data["no"]]
                            for res in results:
                                res['data_task_common'] = data['data']

        return data_list
    def get_cam_source_mapping(self, root_directory):
        mapping = {}
        if not os.path.isdir(root_directory):
            print(f"Error: Root directory '{root_directory}' not found.")
            return mapping

        # When using CLIP features, we have subdirectories
        # When using raw images, we might have .npy files or subdirectories
        for item in sorted(os.listdir(root_directory)):
            video_no = item[:6]
            if not video_no.isdigit() or int(video_no) not in self.split_ids:
                continue
            
            if video_no not in mapping:
                mapping[video_no] = {}
            
            full_path = os.path.join(root_directory, item)
            
            if "headcam" in item:
                main_sub = "Main" if "_Main_" in item else "Sub"
                mapping[video_no][main_sub] = full_path
            elif "envcam" in item:
                match_envcam = re.search(r'envcam(\d)', item)
                if match_envcam:
                    envcam_no = int(match_envcam.group(1))
                    mapping[video_no][f'images{envcam_no}'] = full_path
        return mapping

    def load_feature(self, path):
        if self.feature_cache is not None and path in self.feature_cache:
            return self.feature_cache[path]
        
        feature = np.load(path)
        if self.feature_cache is not None:
            self.feature_cache[path] = feature
        return feature

    def resolve_frame_path(self, dir_path, idx):
        """Resolves the exact frame path, handling the fallback to first/last if missing."""
        f_path = os.path.join(dir_path, f"frame_{idx:06d}.npy")
        if os.path.exists(f_path):
            return f_path
        
        # Fallback to first/last if the index is out of bounds
        if dir_path not in self.dir_cache:
            all_frames = sorted(glob.glob(os.path.join(dir_path, "frame_*.npy")))
            self.dir_cache[dir_path] = all_frames
        else:
            all_frames = self.dir_cache[dir_path]
            
        if not all_frames:
            return None
            
        return all_frames[0] if idx == 0 else all_frames[-1]

    def pre_load_features(self):
        print("Pre-loading CLIP features into cache...")
        paths_to_load = set()
        
        # 1. Collect all required paths
        for motion in tqdm(self.motion_clip):
            if self.use_headcam and 'headcam_info' in motion:
                info = motion['headcam_info']
                vid = info['video_no']
                ms = info['main_sub']
                if vid in self.headcam_source and ms in self.headcam_source[vid]:
                    path = self.headcam_source[vid][ms]
                    idx_s, idx_e = info['indices']
                    p1 = self.resolve_frame_path(path, idx_s)
                    p2 = self.resolve_frame_path(path, idx_e)
                    if p1: paths_to_load.add(p1)
                    if p2: paths_to_load.add(p2)
            
            if self.use_envcam and 'envcam_info' in motion:
                info = motion['envcam_info']
                vid = info['video_no']
                idx_s, idx_e = info['indices']
                for i in range(4):
                    if vid in self.envcam_source and f'images{i}' in self.envcam_source[vid]:
                        path = self.envcam_source[vid][f'images{i}']
                        p1 = self.resolve_frame_path(path, idx_s)
                        p2 = self.resolve_frame_path(path, idx_e)
                        if p1: paths_to_load.add(p1)
                        if p2: paths_to_load.add(p2)
        
        # 2. Bulk load features
        for path in tqdm(list(paths_to_load), desc="Loading features"):
            self.load_feature(path)
        
        print(f"Pre-loaded {len(self.feature_cache)} features into memory.")

    # Legacy image loading methods removed (no longer used)


    def extract_motion_clip(self, mode, motion, annotation,
                            fps, envcam_fps, headcam_fps, max_len, device):
        assert len(annotation) == len(motion), (
            f"motion length:{len(motion)} is not the same to annotation length:{len(annotation)}"
        )

        motion_list = []
        annotation_list = []

        for ann, mo in tzip(annotation, motion):
            assert ann["no"] == mo["no"]
            assert ann["main_sub"] == mo["main_sub"]

            video_no_str = f"{ann['no']:06d}"

            for act in ann["data"]:
                start_str = act["s_time"]
                end_str = act["e_time"]
                hours, minutes, seconds = start_str.split(":")
                start_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                hours, minutes, seconds = end_str.split(":")
                end_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                s_frame = int(start_sec * fps)
                e_frame = int(end_sec * fps)

                base_adic = {
                    "no": ann["no"],
                    "shelf_id": ann["shelf_id"],
                    "ass_dis": ann["ass_dis"],
                    "main_sub": ann["main_sub"],
                    "caption": f'{act["action_verb"]} {act["action_noun"]}',
                    "caption_verb": act["action_verb"],
                }

                base_mdic = {
                    "no": mo["no"],
                    "main_sub": mo["main_sub"]
                }

                if mode in ["action_task", "action_task_taskcommon"]:
                    for task in ann["data_task"]:
                        start_str = task["s_time"]
                        hours, minutes, seconds = start_str.split(":")
                        start_sec_task = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        end_str = act["e_time"]
                        hours, minutes, seconds = end_str.split(":")
                        end_sec_task = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        if start_sec_task <= start_sec and start_sec < end_sec_task:
                            base_adic["caption"] = f'{base_adic["caption"]} to {task["task"]}'
                            # base_adic["caption"] = f'{act["action_verb"]} {act["action_noun"]} to {task["task"]}'
                            break

                if mode in ["action_taskcommon", "action_task_taskcommon"]:
                    for task in ann["data_task_common"]:
                        start_str = task["s_time"]
                        hours, minutes, seconds = start_str.split(":")
                        start_sec_task = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        end_str = act["e_time"]
                        hours, minutes, seconds = end_str.split(":")
                        end_sec_task = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        if start_sec_task <= start_sec and start_sec < end_sec_task:
                            base_adic["caption"] = f'{base_adic["caption"]} to {task["task"]}'
                            # base_adic["caption"] = f'{act["action_verb"]} {act["action_noun"]} to {task["task"]}'
                            break

                # Logic for prediction mode
                if self.task == 'prediction':
                    input_size_frames = int(self.input_seconds * fps)
                    prediction_size_frames = int(self.prediction_seconds * fps)
                    window_size_frames = input_size_frames + prediction_size_frames
                    stride_frames = int(self.stride * fps)
                    
                    action_len = e_frame - s_frame + 1

                    if action_len <= input_size_frames:
                        continue # Skip actions shorter than input length

                    if not self.is_autoregressive:
                        # Sliding window
                        for current_start in range(s_frame, e_frame - input_size_frames + 1, stride_frames):
                            current_end = current_start + window_size_frames - 1

                            real_end = min(current_end, e_frame)
                            valid_length = real_end - current_start + 1
                            pad_length = window_size_frames - valid_length

                            mdic = copy.deepcopy(base_mdic)

                            # Extract the valid portion
                            body_pose = mo["body_pose"][current_start : real_end + 1]
                            global_orient = mo["global_orient"][current_start : real_end + 1]
                            left_hand_pose = mo["left_hand_pose"][current_start : real_end + 1]
                            right_hand_pose = mo["right_hand_pose"][current_start : real_end + 1]
                            root_pos = mo["root_pos"][current_start : real_end + 1]

                            # Pad with zeros if necessary along the frame dimension
                            def pad_tensor(t):
                                if pad_length > 0:
                                    padding = [0, 0] * (t.dim() - 1) + [0, pad_length]
                                    return F.pad(t, padding, mode='constant', value=0)
                                return t

                            mdic["body_pose"] = pad_tensor(body_pose)
                            mdic["global_orient"] = pad_tensor(global_orient)
                            mdic["left_hand_pose"] = pad_tensor(left_hand_pose)
                            mdic["right_hand_pose"] = pad_tensor(right_hand_pose)
                            mdic["root_pos"] = pad_tensor(root_pos)

                            # Extract envcam image metadata
                            if self.use_envcam:
                                mdic["envcam_info"] = {
                                    'video_no': video_no_str,
                                    'indices': [min(int(current_start/fps*envcam_fps), 1000000), 
                                                min(int((real_end+1)/fps*envcam_fps), 1000000)] # placeholder, will cap at real len in getitem
                                }

                            # Extract headcam image metadata
                            if self.use_headcam:
                                mdic["headcam_info"] = {
                                    'video_no': video_no_str,
                                    'main_sub': ann["main_sub"],
                                    'indices': [min(int(current_start/fps*headcam_fps), 1000000), 
                                                min(int((real_end+1)/fps*headcam_fps), 1000000)]
                                }

                            motion_list.append(mdic)

                            adic = copy.deepcopy(base_adic)
                            # Store timing info for debugging/reference
                            adic["clip_start_frame"] = current_start
                            adic["clip_end_frame"] = real_end
                            adic["valid_length"] = valid_length
                            adic["pad_length"] = pad_length
                            annotation_list.append(adic)

                        continue # Done with this action for prediction mode

                # Normal mode logic (unchanged)
                if not self.is_autoregressive and s_frame + max_len - 1 < e_frame:
                    e_frame = s_frame + max_len - 1

                mdic = copy.deepcopy(base_mdic)
                mdic["body_pose"] = mo["body_pose"][
                    s_frame : e_frame + 1
                ]  # .to(device)
                mdic["global_orient"] = mo["global_orient"][
                    s_frame : e_frame + 1
                ]  # .to(device)
                mdic["left_hand_pose"] = mo["left_hand_pose"][
                    s_frame : e_frame + 1
                ]  # .to(device)
                mdic["right_hand_pose"] = mo["right_hand_pose"][
                    s_frame : e_frame + 1
                ]  # .to(device)
                mdic["root_pos"] = mo["root_pos"][s_frame : e_frame + 1]  # .to(device)

                if self.use_envcam:
                    mdic["envcam_info"] = {
                        'video_no': video_no_str,
                        'indices': [int(s_frame/fps*envcam_fps), int(e_frame/fps*envcam_fps)]
                    }

                if self.use_headcam:
                    mdic["headcam_info"] = {
                        'video_no': video_no_str,
                        'main_sub': ann["main_sub"],
                        'indices': [int(s_frame/fps*headcam_fps), int(e_frame/fps*headcam_fps)]
                    }

                motion_list.append(mdic)

                adic = copy.deepcopy(base_adic)
                adic["valid_length"] = e_frame - s_frame + 1
                annotation_list.append(adic)

        return motion_list, annotation_list

    def _organize_joint_motion_pairs(self, motion_list, annotation_list):
        """
        Reorganize motion and annotation lists to create pairs of (Main, Sub) motions.
        For each action, create a sample where Main motion is conditioning and Sub motion is target.
        
        :param motion_list: List of motion dictionaries with 'no' and 'main_sub' fields
        :param annotation_list: List of annotation dictionaries with 'no' and 'main_sub' fields
        :return: Reorganized (motion_list, annotation_list) with paired Main/Sub samples
        """
        # Create mappings by (no, clip_start_frame) to handle multiple clips per action
        main_motions = {}
        sub_motions = {}
        main_annotations = {}
        sub_annotations = {}
        
        for i, (motion, annotation) in enumerate(zip(motion_list, annotation_list)):
            no = motion['no']
            main_sub = motion['main_sub']
            # Use clip_start_frame if available (for prediction mode), else use 0
            clip_key = motion.get('clip_start_frame', 0)
            pair_key = (no, clip_key)
            
            if main_sub == 'Main':
                main_motions[pair_key] = motion
                main_annotations[pair_key] = annotation
            elif main_sub == 'Sub':
                sub_motions[pair_key] = motion
                sub_annotations[pair_key] = annotation
        
        # Create paired samples
        paired_motions = []
        paired_annotations = []
        
        for pair_key in main_motions.keys():
            if pair_key in sub_motions:
                # Create a combined motion dict with both Main (as conditioning) and Sub (as target)
                combined_motion = {
                    'no': main_motions[pair_key]['no'],
                    'main_sub': 'Main_Sub_Pair',
                    'main_motion': main_motions[pair_key],  # Main person's motion (conditioning)
                    'sub_motion': sub_motions[pair_key],    # Sub person's motion (target to predict)
                }
                
                # Create combined annotation with both
                combined_annotation = {
                    'no': main_annotations[pair_key]['no'],
                    'main_sub': 'Main_Sub_Pair',
                    'caption': main_annotations[pair_key].get('caption', 'paired_motion'),
                    'main_annotation': main_annotations[pair_key],
                    'sub_annotation': sub_annotations[pair_key],
                }
                
                # Copy key fields from main annotation
                for key in ['shelf_id', 'ass_dis', 'valid_length', 'clip_start_frame', 'clip_end_frame', 'pad_length']:
                    if key in main_annotations[pair_key]:
                        combined_annotation[key] = main_annotations[pair_key][key]
                
                paired_motions.append(combined_motion)
                paired_annotations.append(combined_annotation)

        return paired_motions, paired_annotations

    def __len__(self):
        return len(self.annotation_clip)

    def __getitem__(self, idx):
        motion = copy.deepcopy(self.motion_clip[idx])
        text = self.annotation_clip[idx]

        # For joint_motion_prediction, process both main and sub motions
        if self.task == 'joint_motion_prediction' and 'main_motion' in motion:
            main_motion = motion['main_motion']
            sub_motion = motion['sub_motion']
            
            # Process main motion features
            main_motion = self._load_motion_features(main_motion)
            # Process sub motion features
            sub_motion = self._load_motion_features(sub_motion)
            
            # Return both motions
            return {'main_motion': main_motion, 'sub_motion': sub_motion}, text
        else:
            # Normal single motion processing
            motion = self._load_motion_features(motion)
            return motion, text

    def _load_motion_features(self, motion):
        """Load camera features for a single motion"""
        # On-demand loading of CLIP features
        if self.use_headcam and 'headcam_info' in motion:
            info = motion['headcam_info']
            vid = info['video_no']
            ms = info['main_sub']
            
            # Default to zero features if missing
            motion['headcam'] = torch.zeros((2, 512))
            
            if vid in self.headcam_source and ms in self.headcam_source[vid]:
                path = self.headcam_source[vid][ms]
                idx_s, idx_e = info['indices']
                
                f1 = self.resolve_frame_path(path, idx_s)
                f2 = self.resolve_frame_path(path, idx_e)
                
                if f1 and f2:
                    motion['headcam'] = torch.from_numpy(np.stack([self.load_feature(f1), self.load_feature(f2)], axis=0))
            del motion['headcam_info']

        if self.use_envcam and 'envcam_info' in motion:
            info = motion['envcam_info']
            vid = info['video_no']
            
            for i in range(4):
                # Default to zero features
                motion[f'envcam{i}'] = torch.zeros((2, 512))
                
                if vid in self.envcam_source and f'images{i}' in self.envcam_source[vid]:
                    path = self.envcam_source[vid][f'images{i}']
                    idx_s, idx_e = info['indices']
                    
                    f1 = self.resolve_frame_path(path, idx_s)
                    f2 = self.resolve_frame_path(path, idx_e)
                    
                    if f1 and f2:
                        motion[f'envcam{i}'] = torch.from_numpy(np.stack([self.load_feature(f1), self.load_feature(f2)], axis=0))
            del motion['envcam_info']
        
        return motion

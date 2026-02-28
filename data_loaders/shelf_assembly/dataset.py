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

        envcam_img = [0] * len(motion_data)
        headcam_img = [0] * len(motion_data)
        if self.use_envcam:
            envcam_img = self.load_envcam_img_npy(self.opt.envcam_dir)
            # duplicate the data for Sub person
            envcam_img = [x for x in envcam_img for _ in range(2)]
        if self.use_headcam:
            headcam_img = self.load_headcam_img_npy(self.opt.headcam_dir)

        self.motion_clip, self.annotation_clip = self.extract_motion_clip(
            mode, motion_data, annotation, envcam_img, headcam_img,
            self.opt.fps, self.opt.envcam_fps, self.opt.headcam_fps,
            self.opt.max_motion_length, device)

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

    def load_headcam_img_npy(self, root_directory):
        all_processed_data = []
        if not os.path.isdir(root_directory):
            print(f"Error: Root directory '{root_directory}' not found.")
            return all_processed_data

        for filename in tqdm(sorted(os.listdir(root_directory)), desc='Loading headcam images'):
            if filename.lower().endswith('.npy'):
                image_array = np.load(os.path.join(root_directory, filename), mmap_mode='r')
                video_no = filename[:6]

                # Filter by split
                if int(video_no) not in self.split_ids:
                    continue

                if "_Main_" in filename:
                    main_sub = "Main"
                elif "_Sub_" in filename:
                    main_sub = "Sub"

                if main_sub is None:
                    print(f"Warning: Could not determine 'Main' or 'Sub' from '{filename}'. Skipping.")
                    continue

                data_entry = {
                    'no': int(video_no),
                    'main_sub': main_sub,
                    'images': image_array
                }
                all_processed_data.append(data_entry)

        return all_processed_data

    def load_headcam_img(self, root_directory):
        # output list[dic]. dic={'no','main_sub','images'}

        all_processed_data = []

        if not os.path.isdir(root_directory):
            print(f"Error: Root directory '{root_directory}' not found.")
            return all_processed_data

        # Get all direct subdirectories
        subdirs = [d for d in sorted(os.listdir(root_directory)) if os.path.isdir(os.path.join(root_directory, d))]

        if not subdirs:
            print(f"No subdirectories found in '{root_directory}'.")
            return all_processed_data

        print(f"Found {len(subdirs)} subdirectories to process.")

        for subdir_name in tqdm(subdirs, desc="Loading headcam images"):
            full_subdir_path = os.path.join(root_directory, subdir_name)

            # Assumes the first 6 characters are always the number
            video_no = subdir_name[:6]

            # Extract 'main_sub' ('Main' or 'Sub')
            main_sub = None
            if "_Main_" in subdir_name:
                main_sub = "Main"
            elif "_Sub_" in subdir_name:
                main_sub = "Sub"

            if main_sub is None:
                print(f"Warning: Could not determine 'Main' or 'Sub' from '{subdir_name}'. Skipping.")
                continue

            image_array = self._load_imgs_in_dir(full_subdir_path)

            data_entry = {
                'no': int(video_no),
                'main_sub': main_sub,
                'images': image_array
            }
            all_processed_data.append(data_entry)

        return all_processed_data

    def load_envcam_img_npy(self, root_directory):
        # output list[dic]. dic={'no','images0','images1','images2','images3'}
        all_processed_data = {}

        if not os.path.isdir(root_directory):
            print(f"Error: Root directory '{root_directory}' not found.")
            return all_processed_data

        for filename in tqdm(sorted(os.listdir(root_directory)), desc='Loading envcam images'):
            if filename.lower().endswith('.npy'):
                video_no = filename[:6]

                # Filter by split
                if int(video_no) not in self.split_ids:
                    continue
                
                match_envcam = re.search(r'envcam(\d)', filename)
                envcam_no = int(match_envcam.group(1))

                if video_no not in all_processed_data:
                    all_processed_data[video_no] = {}

                image_array = np.load(os.path.join(root_directory, filename), mmap_mode='r')
                all_processed_data[video_no][f'images{envcam_no}'] = image_array

                if 'no' not in all_processed_data[video_no]:
                    all_processed_data[video_no]['no'] = int(video_no)

        sorted_keys = sorted(all_processed_data.keys(), key=lambda k: int(k))

        all_processed_data_list = []
        for key in sorted_keys:
            all_processed_data_list.append(all_processed_data[key])

        return all_processed_data_list

    def load_envcam_img(self, root_directory):
        # output list[dic]. dic={'no','images0','images1','images2','images3'}
        all_processed_data = {}

        if not os.path.isdir(root_directory):
            print(f"Error: Root directory '{root_directory}' not found.")
            return all_processed_data

        # Get all direct subdirectories
        subdirs = [d for d in sorted(os.listdir(root_directory)) if os.path.isdir(os.path.join(root_directory, d))]

        if not subdirs:
            print(f"No subdirectories found in '{root_directory}'.")
            return all_processed_data

        print(f"Found {len(subdirs)} subdirectories to process.")

        for subdir_name in tqdm(subdirs, desc="Loading envcam images"):
            full_subdir_path = os.path.join(root_directory, subdir_name)
            # Assumes the first 6 characters are always the number
            video_no = subdir_name[:6]
            match_envcam = re.search(r'envcam(\d)', subdir_name)
            envcam_no = int(match_envcam.group(1))

            if video_no not in all_processed_data:
                all_processed_data[video_no] = {}

            image_array = self._load_imgs_in_dir(full_subdir_path)
            all_processed_data[video_no][f'images{envcam_no}'] = image_array

            if 'no' not in all_processed_data[video_no]:
                all_processed_data[video_no]['no'] = int(video_no)

        sorted_keys = sorted(all_processed_data.keys(), key=lambda k: int(k))

        all_processed_data_list = []
        for key in sorted_keys:
            all_processed_data_list.append(all_processed_data[key])

        return all_processed_data_list

    def _load_imgs_in_dir(self, dir_path):
        image_files = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.lower().startswith('frame_') and filename.lower().endswith('.jpg'):
                # Extract frame number for sorting
                # Assuming format 'frame_XXXXXX.jpg'
                match = re.match(r'frame_(\d{6})\.jpg', filename, re.IGNORECASE)
                if match:
                    frame_number = int(match.group(1))
                    image_files.append((frame_number, os.path.join(dir_path, filename)))

        # Sort image files by their frame number
        image_files.sort(key=lambda x: x[0])

        # 4. Load images and concatenate
        concatenated_images = []
        if not image_files:
            # print(f"Warning: No image files found in '{full_subdir_path}'. 'image' will be an empty array.")
            # If no images, we'll just add an empty numpy array for 'image'
            image_array = np.empty((0, 0, 0, 3), dtype=np.uint8) # Default empty shape for consistency
        else:
            # Load images
            for _, img_path in image_files:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image '{img_path}'. Skipping.")
                    continue
                concatenated_images.append(img)

            if concatenated_images:
                image_array = np.stack(concatenated_images, axis=0) # Concatenate along a new 0th (time) dimension
            else:
                image_array = np.empty((0, 0, 0, 3), dtype=np.uint8) # Fallback if all images failed to load

        return image_array


    def extract_motion_clip(self, mode, motion, annotation, envcam, headcam,
                            fps, envcam_fps, headcam_fps, max_len, device):
        assert len(annotation) == len(motion), (
            f"motion length:{len(motion)} is not the same to annotation length:{len(annotation)}"
        )
        if self.use_envcam:
            assert len(annotation) == len(envcam), (
            f"envcam data length:{len(envcam)} is not the same to annotation length:{len(annotation)}"
        )
        if self.use_headcam:
            assert len(annotation) == len(headcam), (
            f"headcam data length:{len(headcam)} is not the same to annotation length:{len(annotation)}"
        )

        motion_list = []
        annotation_list = []

        if self.task == 'prediction':
            window_size_frames = int((self.input_seconds + self.prediction_seconds) * fps)
            stride_frames = int(self.stride * fps)

        for ann, mo, env, head in tzip(annotation, motion, envcam, headcam):
            assert ann["no"] == mo["no"]
            assert ann["main_sub"] == mo["main_sub"]

            if self.use_envcam:
                assert ann["no"] == env["no"]

            if self.use_headcam:
                assert ann["no"] == head["no"]
                assert ann["main_sub"] == head["main_sub"]

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

                            # Extract envcam image data
                            if self.use_envcam:
                                ec_start = min(int(current_start/fps*envcam_fps), len(env)-2)
                                ec_end   = min(int((real_end+1)/fps*envcam_fps), len(env)-1)
                                indices = [ec_start, ec_end]
                                for i in range(4):  # Assuming keys follows the pattern 'images0', 'images1', ...
                                    key = f"images{i}"
                                    if key in env:
                                        images = np.stack([env[key][k] for k in indices], axis=0)
                                        mdic[f"envcam{i}"] = torch.from_numpy(images)

                            # Extract headcam image data
                            if self.use_headcam:
                                he_start = min(int(current_start/fps*headcam_fps), len(head)-2)
                                he_end   = min(int((real_end+1)/fps*headcam_fps), len(head)-1)
                                indices = [he_start, he_end]
                                images = np.stack([head["images"][k] for k in indices], axis=0)
                                mdic["headcam"] = torch.from_numpy(images)

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
                    ec_start = min(int(current_start/fps*envcam_fps), len(envcam)-2)
                    ec_end   = min(int((real_end+1)/fps*envcam_fps), len(envcam)-1)
                    indices = [ec_start, ec_end]
                    for i in range(4):  # Assuming keys follows the pattern 'images0', 'images1', ...
                        key = f"images{i}"
                        if key in env:
                            images = np.stack([env[key][k] for k in indices], axis=0)
                            mdic[f"envcam{i}"] = torch.from_numpy(images)

                if self.use_headcam:
                    ec_start = min(int(current_start/fps*headcam_fps), len(headcam)-2)
                    ec_end   = min(int((real_end+1)/fps*headcam_fps), len(headcam)-1)
                    indices = [ec_start, ec_end]
                    images = np.stack([head["images"][k] for k in indices], axis=0)
                    mdic["headcam"] = torch.from_numpy(images)

                motion_list.append(mdic)

                adic = copy.deepcopy(base_adic)
                adic["valid_length"] = e_frame - s_frame + 1
                annotation_list.append(adic)

        return motion_list, annotation_list

    def __len__(self):
        return len(self.annotation_clip)

    def __getitem__(self, idx):
        motion = self.motion_clip[idx]
        text = self.annotation_clip[idx]
        return motion, text

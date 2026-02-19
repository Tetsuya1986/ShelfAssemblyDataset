import copy
import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from data_loaders.humanml.utils.get_opt import get_opt
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d


class ShelfAssemblyDataset(data.Dataset):
    def __init__(
        self, mode, datapath="./dataset/shelf_assembly_opt.txt", split="train", **kwargs
    ):
        abs_base_path = kwargs.get("abs_path", ".")
        dataset_opt_path = os.path.join(abs_base_path, datapath)
        device = kwargs.get("device", None)
        opt = get_opt(dataset_opt_path, device)
        max_frames = opt.max_motion_length

        print(f"Loading dataset {opt.dataset_name} ...")
        self.motion_data = self.load_motion_data(opt.motion_dir, mode, device)
        self.annotation = self.load_annotation(opt.text_dir, mode, device)
        # headcam_img = self.load_headcam_img(opt.headcam_dir, mode, annotation, max_frames, device)
        fps = opt.fps
        max_len = opt.max_motion_length
        self.motion_clip, self.annotation_clip = self.extract_motion_clip(
            mode, self.motion_data, self.annotation, fps, max_len, device
        )

    def load_motion_data(self, motion_dir, mode, device):
        data_list = []
        file_pattern = os.path.join(motion_dir, "*.npz")
        files = sorted(glob.glob(file_pattern))
        
        for filepath in tqdm(files, desc="Loading motion data"):
            filename = os.path.basename(filepath)
            if mode == "action":
                # Check filename pattern if necessary (e.g. HH check)
                if filename[17:19] == "HH":
                    no = int(filename[:6])
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
        data_list = []
        for root, _, files in os.walk(text_dir):
            sorted_files = sorted(files)
            for filename in sorted_files:
                if mode == "action":
                    if (
                        filename.lower().endswith("_action.json")
                        and filename[17:19] == "HH"
                    ):
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
        return data_list

    def load_headcam_img(self, headcam_dir, mode, annotation, max_frames, device):
        data_list = []
        idx = 0
        for root, _, files in os.walk(headcam_dir):
            sorted_files = sorted(files)
            for filename in tqdm(sorted_files, desc="files"):
                if mode == "action":
                    if (
                        filename.lower().endswith("headcam.mp4")
                        and int(filename[:6]) == annotation[idx]["no"]
                        and filename[11:14] == annotation[idx]["ass_dis"][:3]
                    ):
                        filepath = os.path.join(root, filename)
                        cap = cv2.VideoCapture(filepath)
                        imgs = []
                        for ann in tqdm(annotation[idx]["data"], desc="ann"):
                            start_str = ann["s_time"]
                            end_str = ann["e_time"]
                            hours, minutes, seconds = start_str.split(":")
                            start_sec = (
                                int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                            )
                            hours, minutes, seconds = end_str.split(":")
                            end_sec = (
                                int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                            )

                            fps = cap.get(cv2.CAP_PROP_FPS)
                            start_frame = int(start_sec * fps)
                            end_frame = int(end_sec * fps)

                            frames = []
                            current_frame = 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    print("break")
                                    break

                                if (
                                    current_frame >= start_frame
                                    and current_frame <= end_frame
                                ):
                                    frames.append(frame)
                                elif current_frame > end_frame:
                                    break

                                current_frame += 1

                            frames_array = np.array(frames)

                            if frames_array.shape[0] > max_frames:
                                frames_array = frames_array[:max_frames, :, :, :]
                            elif frames_array.shape[0] < max_frames:
                                T, X, Y, C = frames_array.shape
                                pad_len = max_frames - T
                                padding = np.zeros(
                                    (pad_len, X, Y, C), dtype=frames_array.dtype
                                )
                                frames_array = np.concatenate(
                                    (frames_array, padding), axis=0
                                )
                            # else:
                            #     do nothing

                            imgs.append(frames_array)

                        cap.release()
                        imgs_array = np.stack(imgs, axis=0)
            data_list.append(imgs_array)

        return data_list

    def extract_motion_clip(self, mode, motion, annotation, fps, max_len, device):
        assert len(annotation) == len(motion), (
            f"motion length:{len(motion)} is not the same to annotation length:{len(annotation)}"
        )

        motion_list = []
        annotation_list = []
        for ann, mo in tqdm(zip(annotation, motion), desc="extract"):
            assert ann["no"] == mo["no"]
            assert ann["main_sub"] == mo["main_sub"]

            for act in ann["data"]:
                start_str = act["s_time"]
                end_str = act["e_time"]
                hours, minutes, seconds = start_str.split(":")
                start_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                hours, minutes, seconds = end_str.split(":")
                end_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                s_frame = int(start_sec * fps)
                e_frame = int(end_sec * fps)

                if s_frame + max_len - 1 < e_frame:
                    e_frame = s_frame + max_len - 1

                mdic = {}
                mdic["no"] = mo["no"]
                mdic["main_sub"] = mo["main_sub"]
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
                motion_list.append(mdic)

                adic = {}
                adic["no"] = ann["no"]
                adic["shelf_id"] = ann["shelf_id"]
                adic["ass_dis"] = ann["ass_dis"]
                adic["main_sub"] = ann["main_sub"]
                adic["caption"] = act["action_verb"] + " " + act["action_noun"]
                adic["caption_verb"] = act["action_verb"]
                annotation_list.append(adic)

        return motion_list, annotation_list

    def __len__(self):
        return len(self.annotation_clip)

    def __getitem__(self, idx):
        motion = self.motion_clip[idx]
        text = self.annotation_clip[idx]
        return motion, text

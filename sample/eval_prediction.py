import os
import numpy as np
import torch
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from utils.sampler_util import PredictionAutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
from tqdm import tqdm
from utils.parser_util import generate_args

def calculate_metrics(gt_motion, pred_motions):
    """
    Calculates ADE, FDE, MPJPE, and APD between a ground-truth motion and K predicted motions.
    gt_motion: [frames, joints, 3] (xyz positions)
    pred_motions: [K, frames, joints, 3]
    """
    K, T, J, _ = pred_motions.shape
    
    # Flatten joints dimension into a single pose vector for ADE/FDE/APD calculation
    # gt_pose: [T, J*3], pred_pose: [K, T, J*3]
    gt_pose = gt_motion.reshape(T, J * 3)
    pred_pose = pred_motions.reshape(K, T, J * 3)
    
    # 1. ADE@K: Average Displacement Error (Root Trajectory)
    # ADE = min_{k} (1/T * \sum_{t=1}^T ||p_{k,t,0} - \hat{p}_{t,0}||_2)
    ade_l2_dist = np.linalg.norm(pred_motions[:, :, 0, :] - gt_motion[None, :, 0, :], axis=-1)  # [K, T]
    ade_per_k = np.mean(ade_l2_dist, axis=1)  # [K]
    ade = np.min(ade_per_k)
    
    # 2. FDE@K: Final Displacement Error (Root Trajectory)
    # FDE = min_{k} ||p_{k,T,0} - \hat{p}_{T,0}||_2
    fde_per_k = np.linalg.norm(pred_motions[:, -1, 0, :] - gt_motion[-1, 0, :], axis=-1)  # [K]
    fde = np.min(fde_per_k)
    
    # 3. MPJPE@K: Mean Per Joint Position Error
    # MPJPE = min_{k} (1/(T*J) * \sum_{t=1}^T \sum_{j=1}^J ||p_{k,t,j} - \hat{p}_{t,j}||_2)
    # Norm is over the 3D coordinate dimension
    mpjpe_l2_dist = np.linalg.norm(pred_motions - gt_motion[None, ...], axis=-1)  # [K, T, J]
    mpjpe_per_k = np.mean(mpjpe_l2_dist, axis=(1, 2))  # [K]
    mpjpe = np.min(mpjpe_per_k)
    
    # 4. APD@K: Average Pairwise Distance (Diversity)
    apd = 0.0
    if K > 1:
        dist_sum = 0.0
        for i in range(K):
            for j in range(K):
                if i != j:
                    # APD = 1/(K*(K-1)) \sum_{i} \sum_{j} (1/T * \sum_{t} ||p_{i,t} - p_{j,t}||_2)
                    p_dist = np.mean(np.linalg.norm(pred_pose[i] - pred_pose[j], axis=-1))
                    dist_sum += p_dist
        apd = dist_sum / (K * (K - 1))
    
    return ade, fde, mpjpe, apd


def calculate_collab_metrics(gt_main_motion, gt_sub_motion, pred_main_motions, pred_sub_motions, 
                             verb, gt_main_root_pos, gt_sub_root_pos, table_height=0.7, hand_joints_indices=None,
                             hold_duration_frames=15, position_stability_threshold=0.01):
    """
    Calculates collaboration-specific metrics between ground-truth and predicted motions for Main and Sub persons.
    
    This function is used for collab_prediction tasks with specific verbs.
    
    Args:
        gt_main_motion: Ground truth Main person motion [frames, joints, 3]
        gt_sub_motion: Ground truth Sub person motion [frames, joints, 3]
        pred_main_motions: Predicted Main person motions [K, frames, joints, 3] or None
        pred_sub_motions: Predicted Sub person motions [K, frames, joints, 3]
        verb: Action verb string. One of: 
              - "hand over", "receive": Measure minimum hand-to-hand distance
              - "pick up", "put down": Measure minimum hand-to-table distance
              - "hold": Measure if Sub person's hand stayed in place for a duration
              - "flip", "rotate": Measure if Main and Sub maintained relative hand position for a duration
        table_height: Height of table surface (y-coordinate). Used for "pick up"/"put down" verbs.
                      If None, will estimate from data.
        hand_joints_indices: Indices of hand joints in the motion data.
                            Default: SMPL left hand (25-39) and right hand (40-54)
        hold_duration_frames: Number of consecutive frames hand must stay in place for "hold" verb
        position_stability_threshold: Maximum allowed positional variation (m) for stability detection
    
    Returns:
        For "hand over"/"receive":
            - min_distance_per_frame: [T] minimum distance between hand pairs
            - mean_min_distance: scalar, mean of min distances
            - min_distance_overall: scalar, minimum distance across all frames
        
        For "pick up"/"put down":
            - min_distance_per_frame: [T] minimum hand-to-table distance per frame
            - mean_min_distance: scalar, mean of min distances
            - min_distance_overall: scalar, minimum distance across all frames
        
        For "hold":
            - stability_ratio: Fraction of frames where hand remained stable
            - mean_hand_stability: Average stability metric per frame
            - num_stable_windows: Number of hold_duration_frames windows with stable position
        
        For "flip"/"rotate":
            - relative_position_stability: Fraction of frames with stable relative position
            - mean_relative_distance_variation: Average frame-to-frame distance variation
            - num_stable_windows: Number of hold_duration_frames windows with stable relative position
    """
    
    # Standard SMPL hand joints: left hand (22-36), right hand (37-52)
    left_hand_joints_indices = list(range(26, 28))
    right_hand_joints_indices = list(range(41, 43))
    
    T = min(gt_main_motion.shape[0], gt_sub_motion.shape[0], pred_sub_motions.shape[1])
    K = pred_sub_motions.shape[0] if pred_sub_motions is not None else 1
    
    verb_lower = verb.lower().strip()
    
    if verb_lower in ["hand over", "receive"]:
        # Calculate minimum distance between Main and Sub person's hands
        
        gt_main_right_hands = gt_main_motion[:, right_hand_joints_indices, :]  # [T, num_hand_joints, 3]
        gt_main_left_hands = gt_main_motion[:, left_hand_joints_indices, :]  # [T, num_hand_joints, 3]
        gt_sub_right_hands = gt_sub_motion[:, right_hand_joints_indices, :]
        gt_sub_left_hands = gt_sub_motion[:, left_hand_joints_indices, :]

        # Ground truth hand distance
        # For each frame, find minimum distance between any hand joint pair
        gt_hand_distances = []
        for t in range(T):
            # Compute distances between all main hand joints and all sub hand joints
            # main_hands[t]: [num_joints, 3], sub_hands[t]: [num_joints, 3]
            main_r_hand = gt_main_right_hands[t]  # [num_joints, 3]
            main_l_hand = gt_main_left_hands[t]  # [num_joints, 3]
            sub_r_hand  = gt_sub_right_hands[t]    # [num_joints, 3]
            sub_l_hand  = gt_sub_left_hands[t]    # [num_joints, 3]

            # Compute pairwise distances: [num_joints, num_joints]
            dis_rr = np.min(np.linalg.norm(main_r_hand[:,  :] - sub_r_hand[:, :], axis=1))
            dis_ll = np.min(np.linalg.norm(main_l_hand[:,  :] - sub_l_hand[:, :], axis=1))
            dis_rl = np.min(np.linalg.norm(main_r_hand[:,  :] - sub_l_hand[:, :], axis=1))
            dis_lr = np.min(np.linalg.norm(main_l_hand[:,  :] - sub_r_hand[:, :], axis=1))
            min_dist = min(dis_rr, dis_ll, dis_rl, dis_lr)

            gt_hand_distances.append(min_dist)

        gt_hand_distances = np.array(gt_hand_distances)  # [T]
        gt_mean_min_distance = np.mean(gt_hand_distances)
        gt_min_distance_overall = np.min(gt_hand_distances)
        # Predicted hand distances
        # If pred_main_motions is provided, use it; otherwise use ground truth Main motion
        
        # Only Sub is predicted, Main is ground truth
        pred_hand_distances_per_k_mean = []
        pred_hand_distances_per_k = []
        for k in range(K):
            pred_sub_right_hands = pred_sub_motions[k, :, right_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_left_hands = pred_sub_motions[k, :, left_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            # pred_sub_right_hands = np.transpose(pred_sub_right_hands, (1, 0, 2)) + gt_sub_root_pos.detach().cpu().numpy()
            # pred_sub_left_hands = np.transpose(pred_sub_left_hands, (1, 0, 2)) + gt_sub_root_pos.detach().cpu().numpy()
            pred_sub_right_hands = np.transpose(pred_sub_right_hands, (1, 0, 2))
            pred_sub_left_hands = np.transpose(pred_sub_left_hands, (1, 0, 2))
            pred_hand_distances = []
            for t in range(T):
                main_r_hand = gt_main_right_hands[t]  # [num_joints, 3]
                main_l_hand = gt_main_left_hands[t]  # [num_joints, 3]
                sub_r_hand = pred_sub_right_hands[t]
                sub_l_hand = pred_sub_left_hands[t]
                dis_rr = np.min(np.linalg.norm(main_r_hand[:,  :] - sub_r_hand[:, :], axis=1))
                dis_ll = np.min(np.linalg.norm(main_l_hand[:,  :] - sub_l_hand[:, :], axis=1))
                dis_rl = np.min(np.linalg.norm(main_r_hand[:,  :] - sub_l_hand[:, :], axis=1))
                dis_lr = np.min(np.linalg.norm(main_l_hand[:,  :] - sub_r_hand[:, :], axis=1))
                min_dist = min(dis_rr, dis_ll, dis_rl, dis_lr)
                pred_hand_distances.append(min_dist)

            pred_hand_distances = np.array(pred_hand_distances)  # [T]
            pred_hand_distances_per_k.append(pred_hand_distances)
            pred_hand_distances_per_k_mean.append(np.mean(pred_hand_distances))

        # Select best prediction
        best_k = np.argmin(pred_hand_distances_per_k_mean)
        pred_sub_hands_best = pred_hand_distances_per_k[best_k]
        
        pred_hand_distances = []
        for t in range(T):
            distances = np.linalg.norm(gt_hand_distances - pred_sub_hands_best)
            min_dist = np.min(distances)
            pred_hand_distances.append(min_dist)
        
        pred_hand_distances = np.array(pred_hand_distances)  # [T]
        pred_mean_min_distance = np.mean(pred_hand_distances)
        pred_min_distance_overall = np.min(pred_hand_distances)
        return {
            'gt_min_distance_per_frame': gt_hand_distances,
            'gt_mean_min_distance': gt_mean_min_distance,
            'gt_min_distance_overall': gt_min_distance_overall,
            'pred_min_distance_per_frame': pred_hand_distances,
            'pred_mean_min_distance': pred_mean_min_distance,
            'pred_min_distance_overall': pred_min_distance_overall,
            'verb': verb,
        }
    
    elif verb_lower in ["pick up", "put down"]:
        # Calculate minimum distance between hand and table
        # If pred_main_motions is None, we use Sub person's hand instead
        
        # # Sub person hand distance to table (for collab_prediction)
        # gt_main_right_hands = gt_main_motion[:, right_hand_joints_indices, :]  # [T, num_hand_joints, 3]
        # gt_main_left_hands = gt_main_motion[:, left_hand_joints_indices, :]  # [T, num_hand_joints, 3]
        # gt_sub_right_hands = gt_sub_motion[:, right_hand_joints_indices, :]
        # gt_sub_left_hands = gt_sub_motion[:, left_hand_joints_indices, :]
        
        # gt_hand_table_distances = []
        
        # for t in range(T):
        #     r_hand = gt_sub_right_hands[t]  # [num_joints, 3]
        #     l_hand = gt_sub_left_hands[t]  # [num_joints, 3]
        #     # Distance from hand to table plane (vertical distance)
        #     r_dis_to_table = np.abs(r_hand[:, 1] - table_height)
        #     l_dis_to_table = np.abs(l_hand[:, 1] - table_height)
        #     min_dist = min(np.min(r_dis_to_table), np.min(l_dis_to_table))
        #     gt_hand_table_distances.append(min_dist)

        # gt_hand_table_distances = np.array(gt_hand_table_distances)  # [T]
        # gt_mean_min_distance = np.mean(gt_hand_table_distances)
        # gt_min_distance_overall = np.min(gt_hand_table_distances)

        # Predicted hand-table distances
        # If pred_main_motions is provided, use it; otherwise use pred_sub_motions

        # Sub person hand distances to table (for collab_prediction where we predict Sub)
        pred_hand_table_distances_per_k_mean = []
        pred_hand_table_distances_per_k = []
        for k in range(K):
            pred_sub_right_hands = pred_sub_motions[k, :, right_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_left_hands = pred_sub_motions[k, :, left_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_right_hands = np.transpose(pred_sub_right_hands, (1, 0, 2)) + gt_sub_root_pos.detach().cpu().numpy()
            pred_sub_left_hands = np.transpose(pred_sub_left_hands, (1, 0, 2)) + gt_sub_root_pos.detach().cpu().numpy()
            pred_distances = []
            for t in range(T):
                sub_r_hand = pred_sub_right_hands[t, :, 1]
                sub_l_hand = pred_sub_left_hands[t, :, 1]
                dis_r = np.min(np.abs(sub_r_hand - table_height))
                dis_l = np.min(np.abs(sub_l_hand - table_height))
                min_dist = min(dis_r, dis_l)
                pred_distances.append(min_dist)

            pred_hand_table_distances_per_k_mean.append(np.mean(pred_distances))
            pred_hand_table_distances_per_k.append(np.array(pred_distances))

        # Select best prediction
        best_k = np.argmin(pred_hand_table_distances_per_k_mean)
        pred_sub_hands_best = pred_hand_table_distances_per_k[best_k]

        pred_hand_table_distances = np.array(pred_sub_hands_best)  # [T]
        pred_mean_min_distance = np.mean(pred_sub_hands_best)
        pred_min_distance_overall = np.min(pred_sub_hands_best)
        return {
            'gt_min_distance_per_frame': 0,
            'gt_mean_min_distance': 0,
            'gt_min_distance_overall': 0,
            'pred_min_distance_per_frame': pred_hand_table_distances,
            'pred_mean_min_distance': pred_mean_min_distance,
            'pred_min_distance_overall': pred_min_distance_overall,
            'table_height': table_height,
            'verb': verb,
        }
    
    elif verb_lower in ["hold"]:
        # Measure if Sub person's hand stays in the same position for hold_duration_frames
        # Extract Sub person's hand positions
        # gt_sub_hands = gt_sub_motion[:, hand_joints_indices, :]  # [T, num_hand_joints, 3]
        
        # # Compute hand center position (average of all hand joints)
        # gt_sub_hand_center = np.mean(gt_sub_hands, axis=1)  # [T, 3]
        
        # # For each frame, compute distance to the mean position to determine stability
        # hand_mean_position = np.mean(gt_sub_hand_center, axis=0)  # [3]
        # distances_from_mean = np.linalg.norm(gt_sub_hand_center - hand_mean_position[np.newaxis, :], axis=1)  # [T]
        
        # # Frame is stable if distance from mean is below threshold
        # stability_per_frame = distances_from_mean < position_stability_threshold
        # stability_ratio = np.sum(stability_per_frame) / T
        # mean_hand_stability = np.mean(distances_from_mean)
        
        # # Count number of stable windows (consecutive hold_duration_frames)
        # num_stable_windows = 0
        # for t in range(T - hold_duration_frames + 1):
        #     window = stability_per_frame[t:t+hold_duration_frames]
        #     if np.all(window):
        #         num_stable_windows += 1
        
        # For predictions, find best among K
        pred_stability_per_k = []
        for k in range(K):
            pred_sub_right_hands = pred_sub_motions[k, :, right_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_left_hands = pred_sub_motions[k, :, left_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_right_hands = np.transpose(pred_sub_right_hands, (1, 0, 2))
            pred_sub_left_hands = np.transpose(pred_sub_left_hands, (1, 0, 2))
            pred_sub_right_hand_center = np.mean(pred_sub_right_hands, axis=1)  # [T, 3]
            pred_sub_left_hand_center = np.mean(pred_sub_left_hands, axis=1)  # [T, 3]
            pred_sub_hand_dist = pred_sub_right_hand_center - pred_sub_left_hand_center
            import pdb; pdb.set_trace()
            for t in range(T):
                pred_hand_mean = np.mean(pred_sub_hand_center, axis=0)  # [3]
                pred_distances = np.linalg.norm(pred_sub_hand_center - pred_hand_mean[np.newaxis, :], axis=1)  # [T]
                pred_stability_per_k.append(np.mean(pred_distances))
        
        best_k = np.argmin(pred_stability_per_k)
        pred_sub_hands = pred_sub_motions[best_k, :, hand_joints_indices, :]
        pred_sub_hand_center = np.mean(pred_sub_hands, axis=1)
        pred_hand_mean = np.mean(pred_sub_hand_center, axis=0)
        pred_distances = np.linalg.norm(pred_sub_hand_center - pred_hand_mean[np.newaxis, :], axis=1)
        pred_stability = pred_distances < position_stability_threshold
        pred_stability_ratio = np.sum(pred_stability) / T
        pred_mean_stability = np.mean(pred_distances)
        
        pred_num_stable_windows = 0
        for t in range(T - hold_duration_frames + 1):
            window = pred_stability[t:t+hold_duration_frames]
            if np.all(window):
                pred_num_stable_windows += 1

        return {
            'gt_stability_ratio': float(stability_ratio),
            'gt_mean_hand_stability': float(mean_hand_stability),
            'gt_num_stable_windows': int(num_stable_windows),
            'pred_stability_ratio': float(pred_stability_ratio),
            'pred_mean_hand_stability': float(pred_mean_stability),
            'pred_num_stable_windows': int(pred_num_stable_windows),
            'hold_duration_frames': hold_duration_frames,
            'verb': verb,
        }
    
    elif verb_lower in ["flip", "rotate"]:
        # Measure if Main and Sub people maintained relative hand position for a duration
        gt_main_right_hands = gt_main_motion[:, right_hand_joints_indices, :]  # [T, num_hand_joints, 3]
        gt_main_left_hands = gt_main_motion[:, left_hand_joints_indices, :]  # [T, num_hand_joints, 3]
        gt_sub_right_hands = gt_sub_motion[:, right_hand_joints_indices, :]
        gt_sub_left_hands = gt_sub_motion[:, left_hand_joints_indices, :]

        # Compute hand center positions
        gt_main_right_hand_center = np.mean(gt_main_right_hands, axis=1)  # [T, 3]
        gt_main_left_hand_center = np.mean(gt_main_left_hands, axis=1)  # [T, 3]
        gt_sub_right_hand_center = np.mean(gt_sub_right_hands, axis=1)    # [T, 3]
        gt_sub_left_hand_center = np.mean(gt_sub_left_hands, axis=1)    # [T, 3]

        # Truncate
        gt_main_right_hand_center = gt_main_right_hand_center[:T,:]
        gt_main_left_hand_center = gt_main_left_hand_center[:T,:]
        gt_sub_right_hand_center = gt_sub_right_hand_center[:T,:]
        gt_sub_left_hand_center = gt_sub_left_hand_center[:T,:]

        # Compute relative position (vector from Sub to Main)
        gt_rel_pos_rr = gt_main_right_hand_center - gt_sub_right_hand_center  # [T, 3]
        gt_rel_pos_ll = gt_main_left_hand_center - gt_sub_left_hand_center  # [T, 3]
        gt_rel_pos_rl = gt_main_right_hand_center - gt_sub_left_hand_center  # [T, 3]
        gt_rel_pos_lr = gt_main_left_hand_center - gt_sub_right_hand_center  # [T, 3]
        
        # Compute relative distance (magnitude)
        gt_rel_dis_rr = np.linalg.norm(gt_rel_pos_rr, axis=1)  # [T]
        gt_rel_dis_ll = np.linalg.norm(gt_rel_pos_ll, axis=1)  # [T]
        gt_rel_dis_rl = np.linalg.norm(gt_rel_pos_rl, axis=1)  # [T]
        gt_rel_dis_lr = np.linalg.norm(gt_rel_pos_lr, axis=1)  # [T]
        
        # Compute frame-to-frame variation in relative position
        gt_rel_dis_var_rr = np.diff(gt_rel_dis_rr, axis=0)  # [T-1]
        gt_rel_dis_var_ll = np.diff(gt_rel_dis_ll, axis=0)  # [T-1]
        gt_rel_dis_var_rl = np.diff(gt_rel_dis_rl, axis=0)  # [T-1]
        gt_rel_dis_var_lr = np.diff(gt_rel_dis_lr, axis=0)  # [T-1]
        
        # Frame is stable if variation from previous frame is small
        gt_stability_per_frame_rr = np.concatenate([[True], gt_rel_dis_var_rr < position_stability_threshold])  # [T]
        gt_stability_per_frame_ll = np.concatenate([[True], gt_rel_dis_var_ll < position_stability_threshold])  # [T]
        gt_stability_per_frame_rl = np.concatenate([[True], gt_rel_dis_var_rl < position_stability_threshold])  # [T]
        gt_stability_per_frame_lr = np.concatenate([[True], gt_rel_dis_var_lr < position_stability_threshold])  # [T]
        gt_stability_per_frame = [gt_stability_per_frame_rr, gt_stability_per_frame_ll, gt_stability_per_frame_rl, gt_stability_per_frame_lr]
        gt_stability_rr = np.sum(gt_stability_per_frame_rr) / T
        gt_stability_ll = np.sum(gt_stability_per_frame_ll) / T
        gt_stability_rl = np.sum(gt_stability_per_frame_rl) / T
        gt_stability_lr = np.sum(gt_stability_per_frame_lr) / T

        best_k = np.argmax([gt_stability_rr, gt_stability_ll, gt_stability_rl, gt_stability_lr])
        gt_stability_per_frame_best = gt_stability_per_frame[best_k]

        # Count stable windows
        gt_num_stable_windows = 0
        for t in range(T - hold_duration_frames + 1):
            window = gt_stability_per_frame_best[t:t+hold_duration_frames]
            if np.all(window):
                gt_num_stable_windows += 1

        # For predictions
        # Only Sub is predicted, Main is ground truth
        for k in range(K):
            pred_sub_right_hands = pred_sub_motions[k, :, right_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_left_hands = pred_sub_motions[k, :, left_hand_joints_indices, :] + gt_sub_root_pos.detach().cpu().numpy()  # [T, num_hand_joints, 3]
            pred_sub_right_hands = np.transpose(pred_sub_right_hands, (1, 0, 2))
            pred_sub_left_hands = np.transpose(pred_sub_left_hands, (1, 0, 2))
            pred_sub_right_hand_center = np.mean(pred_sub_right_hands, axis=1)  # [T, 3]
            pred_sub_left_hand_center = np.mean(pred_sub_left_hands, axis=1)  # [T, 3]

            pred_rel_pos_rr = gt_main_right_hand_center - pred_sub_right_hand_center  # [T, 3]
            pred_rel_pos_ll = gt_main_left_hand_center - pred_sub_left_hand_center  # [T, 3]
            pred_rel_pos_rl = gt_main_right_hand_center - pred_sub_left_hand_center  # [T, 3]
            pred_rel_pos_lr = gt_main_left_hand_center - pred_sub_right_hand_center  # [T, 3]

            pred_rel_dis_rr = np.linalg.norm(pred_rel_pos_rr, axis=1)  # [T]
            pred_rel_dis_ll = np.linalg.norm(pred_rel_pos_ll, axis=1)  # [T]
            pred_rel_dis_rl = np.linalg.norm(pred_rel_pos_rl, axis=1)  # [T]
            pred_rel_dis_lr = np.linalg.norm(pred_rel_pos_lr, axis=1)  # [T]

            pred_rel_dis_var_rr = np.diff(pred_rel_dis_rr, axis=0)  # [T-1]
            pred_rel_dis_var_ll = np.diff(pred_rel_dis_ll, axis=0)  # [T-1]
            pred_rel_dis_var_rl = np.diff(pred_rel_dis_rl, axis=0)  # [T-1]
            pred_rel_dis_var_lr = np.diff(pred_rel_dis_lr, axis=0)  # [T-1]

            pred_stability_per_frame_rr = np.concatenate([[True], pred_rel_dis_var_rr < position_stability_threshold])  # [T]
            pred_stability_per_frame_ll = np.concatenate([[True], pred_rel_dis_var_ll < position_stability_threshold])  # [T]
            pred_stability_per_frame_rl = np.concatenate([[True], pred_rel_dis_var_rl < position_stability_threshold])  # [T]
            pred_stability_per_frame_lr = np.concatenate([[True], pred_rel_dis_var_lr < position_stability_threshold])  # [T]
            pred_stability_per_frame = [pred_stability_per_frame_rr, pred_stability_per_frame_ll, pred_stability_per_frame_rl, pred_stability_per_frame_lr]

            pred_stability_rr = np.sum(pred_stability_per_frame_rr) / T
            pred_stability_ll = np.sum(pred_stability_per_frame_ll) / T
            pred_stability_rl = np.sum(pred_stability_per_frame_rl) / T
            pred_stability_lr = np.sum(pred_stability_per_frame_lr) / T

            best_k = np.argmax([pred_stability_rr, pred_stability_ll, pred_stability_rl, pred_stability_lr])
            pred_stability_per_frame_best = pred_stability_per_frame[best_k]
            pred_stability_per_frame_best_mean = np.mean(pred_stability_per_frame[best_k])

        pred_num_stable_windows = 0
        for t in range(T - hold_duration_frames + 1):
            window = pred_stability_per_frame_best[t:t+hold_duration_frames]
            if np.all(window):
                pred_num_stable_windows += 1

        return {
            'pred_stability_per_frame_best_mean': pred_stability_per_frame_best_mean,
            'pred_relative_position_stability': pred_stability_per_frame_best,
            'pred_num_stable_windows': int(pred_num_stable_windows),
            'hold_duration_frames': hold_duration_frames,
            'verb': verb,
        }

    else:
        return None

def judge_success_failure(res_dic):
    verb = res_dic['verb']
    if verb in ["hand over", "receive"]:
        print(f"[hand over / receive] {res_dic['pred_min_distance_overall']}")
        return res_dic['pred_min_distance_overall'] < 0.3

    elif verb in ["pick up", "put down"]:
        print(f"[pick up / put down] {res_dic['pred_min_distance_overall']}")
        return res_dic['pred_min_distance_overall'] < 0.05

    elif verb in ["hold"]:
        print(f"[hold] {res_dic['stability_ratio']}")
        return res_dic['stability_ratio'] < 0.1

    elif verb in ["flip", "rotate"]:
        print(f"[flip / rotate] {res_dic['pred_stability_per_frame_best_mean']}")
        print(f"[flip / rotate] {res_dic['hold_duration_frames']}")
        return res_dic['hold_duration_frames'] > 5

    else:
        assert(False)

    return False

def main(args=None):
    if args is None:
        args = generate_args()
        
    # generate_args() / apply_rules() may strip prediction parameters if it loaded `task=generation` from the checkpoint args.json
    # We must explicitly force prediction mode here to evaluate futures.
    args.task = 'collab_prediction'
    # args.task = 'prediction'
    if args.input_seconds is None:
        args.input_seconds = getattr(args, 'input_seconds_cli', 0.5)
    if args.prediction_seconds is None:
        args.prediction_seconds = getattr(args, 'prediction_seconds_cli', 1.0)
    if args.stride is None:
        args.stride = getattr(args, 'stride_cli', 0.5)

    # Force batch size to 1 for evaluation since prediction length will vary per sample
    args.batch_size = 1

    if args.dataset == 'kit':
        fps = 12.5
    elif args.dataset in ['shelf_assembly', 'comad', 'core4d']:
        fps = 30.0
    else:
        fps = 20.0
        
    local_pred_len = int(args.prediction_seconds * fps)
    
    # Do not set args.context_len > 0 or args.pred_len > 0 as it triggers legacy 'prefix' completion in MDM.
    history_len = int(args.input_seconds * fps)

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    print("Loading test dataset...")
    # Load dataset loader for 'test' split
    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=int(args.motion_length * fps),
                              fixed_len=local_pred_len + history_len, 
                              pred_len=local_pred_len,
                              device=dist_util.dev(),
                              task=args.task,
                              input_seconds=args.input_seconds,
                              prediction_seconds=args.prediction_seconds,
                              stride=args.stride,
                              split='test',
                              autoregressive=args.autoregressive,
                              use_envcam=args.use_envcam,
                              use_headcam=args.use_headcam,
                              label_option=args.label_option,
                              data_sel=args.data_sel,
                              )

    print(f"Creating MDM model and diffusion... context_len={args.context_len}, pred_len={args.pred_len}")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.eval()

    if args.model_path != "":
        print(f"Loading checkpoint from {args.model_path}")
        from utils.model_util import load_saved_model
        load_saved_model(model, args.model_path)

    sample_fn = diffusion.p_sample_loop

    fps = getattr(data.dataset, 'fps', 30.0)
    history_len = int(args.input_seconds * fps)
    
    print(f"Input seconds: {args.input_seconds}, FPS: {fps}, History frames: {history_len}")

    if args.autoregressive:
        # AR Sampler internally relies on args.pred_len to evaluate step sizes
        args.pred_len = local_pred_len
        
        sampler = PredictionAutoRegressiveSampler(
            args=args,
            sample_fn=sample_fn, 
            required_frames=local_pred_len,
            history_len=history_len
        )
        sample_fn = sampler.sample

    all_ade = []
    all_fde = []
    all_mpjpe = []
    all_apd = []
    
    # Collaboration metrics tracking
    all_collab_metrics = []
    all_collab_success = []
    handover_success = []
    pickupdown_success = []
    hold_success = []
    flip_rotate_success = []
    is_collab_task = False

    # Support debug subsets
    max_eval_samples = getattr(args, 'num_eval_samples', None)
    if max_eval_samples is not None:
        print(f"DEBUG: Restricting evaluation to {max_eval_samples} samples.")

    print(f"Evaluating batches on test split...")
    
    # Turn off gradients for evaluation
    with torch.no_grad():
        evaluated_count = 0
        for input_motion, model_kwargs in tqdm(data, desc="Evaluating"):
            if max_eval_samples is not None and evaluated_count >= max_eval_samples:
                break
            # Shape of input_motion: [bs, njoints, nfeats, total_frames]
            input_motion = input_motion.to(dist_util.dev())
            model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

            # # debug
            # if model_kwargs['y']['verb'][0] not in ["flip", "rotate"]:
            #     continue

            bs = input_motion.shape[0]
            assert bs == 1
            
            # Total lengths of each sequence
            lengths = model_kwargs['y']['lengths']
            if lengths < history_len:
                continue
            
            # Detect collaboration prediction task
            is_collab_task = hasattr(args, 'task') and args.task == 'collab_prediction'
            verb = model_kwargs['y'].get('verb', None) if 'y' in model_kwargs else None
            if verb:
                verb = verb[0]

            # Extract Main motion for collaboration tasks
            main_motion = None
            if is_collab_task and 'main_motion' in model_kwargs['y']:
                main_motion = model_kwargs['y']['main_motion'].to(dist_util.dev())

            # Set up history condition
            model_kwargs['y']['history'] = input_motion[..., :history_len]
            
            # Predict the entire remaining ground truth sequence natively through autoregressive overlaps
            capped_lengths = torch.clamp(lengths - history_len, min=0)
            model_kwargs['y']['lengths'] = capped_lengths
            
            if 'text' in model_kwargs['y'].keys():
                model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

            # Store predictions for the current batch
            K = args.num_repetitions
            
            # Repeat tensors along batch dimension for batched K inference
            # We assume bs = 1 (as forced above), so after repeat bs = K
            motion_shape = (bs * K, model.njoints, model.nfeats, local_pred_len)
            
            # Duplicate kwargs along batch dim
            batched_kwargs = {'y': {}}
            for key, val in model_kwargs['y'].items():
                if torch.is_tensor(val):
                    # Repeat along the first dimension (batch)
                    # e.g. text_embed: [bs, seq_len, dim] -> [bs*K, seq_len, dim]
                    repeats = [K] + [1] * (val.dim() - 1)
                    batched_kwargs['y'][key] = val.repeat(*repeats)
                elif isinstance(val, list):
                    # E.g. text lists
                    batched_kwargs['y'][key] = val * K
                else:
                    batched_kwargs['y'][key] = val

            # Generate sample in one batched pass
            # Output sample is [bs * K, njoints, nfeats, generated_len]
            batched_sample = sample_fn(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=batched_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            
            batched_sample = batched_sample.to(dist_util.dev())

            # [K, bs, njoints, nfeats, gen_len]
            # Since bs=1, we reshape directly to [K, bs, njoints, nfeats, gen_len]
            batch_predictions = batched_sample.view(K, bs, model.njoints, model.nfeats, -1)

            # Convert to xyz coordinates for evaluation
            # For Sub motion: extract prediction window from input_motion
            gt_max_len = int(lengths.max().item())
            gt_sub_pred_window = input_motion[..., history_len:gt_max_len] # [bs, nj, nfeats, slen]

            if gt_sub_pred_window.shape[-1] == 0:
                continue
            
            # Use model.rot2xyz to get the xyz joint positions
            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask_gt = None if rot2xyz_pose_rep == 'xyz' else torch.ones((bs, gt_sub_pred_window.shape[-1]), dtype=torch.bool, device=dist_util.dev())

            gt_sub_xyz = model.rot2xyz(x=gt_sub_pred_window, mask=rot2xyz_mask_gt, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=True,
                                       get_rotations_back=False) # [bs, nj, 3, slen]
                                   
            # gt_sub_xyz: [bs, nj, 3, slen] -> [bs, slen, nj, 3]
            gt_sub_xyz_np = gt_sub_xyz.permute(0, 3, 1, 2).cpu().numpy()
            
            # For Main motion (collaboration tasks only)
            gt_main_xyz_np = None
            if is_collab_task and main_motion is not None:
                # Extract Main motion prediction window
                gt_main_pred_window = main_motion[..., history_len:gt_max_len]  # [bs, nj, nfeats, slen]
                rot2xyz_mask_main = None if rot2xyz_pose_rep == 'xyz' else torch.ones((bs, gt_main_pred_window.shape[-1]), dtype=torch.bool, device=dist_util.dev())
                
                gt_main_xyz = model.rot2xyz(x=gt_main_pred_window, mask=rot2xyz_mask_main, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=True,
                                       get_rotations_back=False) # [bs, nj, 3, slen]
                
                # gt_main_xyz: [bs, nj, 3, slen] -> [bs, slen, nj, 3]
                gt_main_xyz_np = gt_main_xyz.permute(0, 3, 1, 2).cpu().numpy()

            # Process predictions chronologically to save memory, one repetition at a time
            pred_xyz_list = []
            for rep_i in range(args.num_repetitions):
                pred_chunk = batch_predictions[rep_i] # [bs, nj, nfeats, slen]
                rot2xyz_mask_pred = None if rot2xyz_pose_rep == 'xyz' else torch.ones((bs, pred_chunk.shape[-1]), dtype=torch.bool, device=dist_util.dev())
                
                pred_xyz_chunk = model.rot2xyz(x=pred_chunk, mask=rot2xyz_mask_pred, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=True,
                                       get_rotations_back=False) # [bs, nj, 3, slen]
                pred_xyz_list.append(pred_xyz_chunk)
                
            # [K, bs, nj, 3, slen]
            pred_xyz = torch.stack(pred_xyz_list, dim=0)

            # pred_xyz: [K, bs, nj, 3, slen] -> [bs, K, slen, nj, 3]
            pred_sub_xyz_np = pred_xyz.permute(1, 0, 4, 2, 3).cpu().numpy()

            # Calculate metrics per sample in the batch
            for i in range(bs):
                seq_len = lengths[i]
                if seq_len <= history_len:
                    continue # No future to predict
                
                # We only evaluate on the predicted future segment
                pred_future_len = seq_len - history_len
                
                # gt_sub_xyz_np is [bs, slen, nj, 3], extract future for Sub person
                gt_future = gt_sub_xyz_np[i, :pred_future_len, :, :] # [T, nj, 3]
                
                # If autoregressive included prefix, then prediction array starts at 0 with history
                if args.autoregressive_include_prefix:
                    pred_future = pred_sub_xyz_np[i, :, history_len:history_len + pred_future_len, :, :]
                else:
                    pred_future = pred_sub_xyz_np[i, :, :pred_future_len, :, :]
                
                # Check shapes match. Autoregressive sampler may generate slightly more frames than requested due to iterations.
                actual_pred_len = min(pred_future.shape[1], gt_future.shape[0])
                gt_future = gt_future[:actual_pred_len, ...]
                pred_future = pred_future[:, :actual_pred_len, ...]

                # Extract exactly the 52 structural joints corresponding to the rotated inputs.
                # SMPL-X output order: 0-21 Body, 22-24 Face, 25-39 Left Hand, 40-54 Right Hand, 55+ Face Contours.
                # The 53 input features are: 1 trans + 52 rotations (22 body, 15 LH, 15 RH).
                if args.dataset in ['shelf_assembly']:
                    core_joints_indices = list(range(22)) + list(range(25, 55))
                elif args.dataset in ['core4d']:
                    core_joints_indices = list(range(22))
                else:
                    core_joints_indices = list(range(gt_future.shape[1]))

                gt_future = gt_future[:, core_joints_indices, :]
                pred_future = pred_future[:, :, core_joints_indices, :]
                
                # Standard prediction metrics
                ade, fde, mpjpe, apd = calculate_metrics(gt_future, pred_future)
                all_ade.append(ade)
                all_fde.append(fde)
                all_mpjpe.append(mpjpe)
                all_apd.append(apd)


                # Calculate metrics based on task type
                if is_collab_task and verb is not None and gt_main_xyz_np is not None:
                    # For collaboration prediction tasks, use Main and Sub motions from data
                    # gt_future (gt_sub): [T, nj, 3] (Sub person ground truth)
                    # pred_future (pred_sub): [K, T, nj, 3] (Sub person predictions)
                    # gt_main_xyz_np from main_motion: [bs, slen, nj, 3]
                    
                    # Extract Main motion for this sample and future window
                    gt_main_future = gt_main_xyz_np[i, :pred_future_len, :, :]  # [T, nj, 3]
                    
                    # Apply core joints indices to Main motion too
                    gt_main_future = gt_main_future[:, core_joints_indices, :]
                    
                    # For predictions, we only predict Sub motion, not Main
                    # Main motion ground truth is used as context
                    # Predictions are for Sub person motion
                    collab_metrics = calculate_collab_metrics(
                        gt_main_motion=gt_main_future,
                        gt_sub_motion=gt_future,
                        pred_main_motions=None,  # No Main predictions, only ground truth
                        pred_sub_motions=pred_future,
                        verb=verb,
                        gt_main_root_pos=model_kwargs['y']['main_motion'][0, 0, :3, 0],
                        gt_sub_root_pos=input_motion[0, 0, :3, 0]
                    )

                    all_collab_metrics.append(collab_metrics)
                    if collab_metrics:
                        success = judge_success_failure(collab_metrics)
                        all_collab_success.append(success)
                        if collab_metrics['verb'] in ["hand over", "receive"]:
                            handover_success.append(success)
                        elif collab_metrics['verb'] in ["pick up", "put down"]:
                            pickupdown_success.append(success)
                        elif collab_metrics['verb'] in ["hold"]:
                            hold_success.append(success)
                        elif collab_metrics['verb'] in ["flip", "rotate"]:
                            flip_rotate_success.append(success)


            evaluated_count += bs

    if is_collab_task:
        all_success_rate = sum(all_collab_success)/len(all_collab_success)
        handover_success_rate = sum(handover_success)/len(handover_success)
        pickupdown_success_rate = sum(pickupdown_success)/len(pickupdown_success)
        # hold_success_rate = sum(hold_success)/len(hold_success)
        flip_rotate_success_rate = sum(flip_rotate_success)/len(flip_rotate_success)

    if len(all_ade) == 0 and len(all_collab_metrics) == 0:
        print("No valid sequences found for evaluation.")
        return

    # Aggregate metrics
    if is_collab_task:
        # Collaboration prediction metrics
        res_str = (
            f"[ALL] Number of Samples: {len(all_collab_success)}, Success Rate: {all_success_rate}\n"
            f"[handover] Number of Samples: {len(handover_success)}, Success Rate: {handover_success_rate}\n"
            f"[pickup] cNumber of Samples: {len(pickupdown_success)}, Success Rate: {pickupdown_success_rate}\n"
            # f"[hold] Number of Samples: {len(hold_success)}, Success Rate: {hold_success_rate}\n"
            f"[rotate] Number of Samples: {len(flip_rotate_success)}, Success Rate: {flip_rotate_success_rate}\n"
        )
        print(res_str)

    # Standard prediction metrics
    mean_ade = np.mean(all_ade)
    mean_fde = np.mean(all_fde)
    mean_mpjpe = np.mean(all_mpjpe)
    mean_apd = np.mean(all_apd)

    res_str = (
        f"Prediction Evaluation Metrics (K={args.num_repetitions})\n"
        f"ADE@{args.num_repetitions}: {mean_ade:.4f}\n"
        f"FDE@{args.num_repetitions}: {mean_fde:.4f}\n"
        f"MPJPE@{args.num_repetitions}: {mean_mpjpe:.4f}\n"
        f"APD@{args.num_repetitions}: {mean_apd:.4f}\n"
    )
    print(res_str)
    
    import os
    from datetime import datetime
    import json
    
    # Create an output directory based on the checkpoint name and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.dirname(args.model_path))
    ckpt_name = os.path.basename(args.model_path).replace('.pt', '')
    
    out_dir_name = f"eval_{ckpt_name}_K{args.num_repetitions}_{args.split}_{timestamp}"
    out_path = os.path.join(os.path.dirname(args.model_path), out_dir_name)
    
    os.makedirs(out_path, exist_ok=True)
    
    # Save the metrics
    metrics_path = os.path.join(out_path, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(res_str)
    
    # Save detailed collaboration metrics if applicable
    # if is_collab_task and len(all_collab_metrics) > 0:
    #     collab_metrics_path = os.path.join(out_path, 'collab_metrics.json')
    #     collab_metrics_summary = {
    #         'verb': all_collab_metrics[0]['verb'],
    #         'num_samples': len(all_collab_metrics),
    #         'mean_pred_distance': float(np.mean([m['pred_mean_min_distance'] for m in all_collab_metrics])),
    #         'mean_gt_distance': float(np.mean([m['gt_mean_min_distance'] for m in all_collab_metrics])),
    #         'mean_min_distance_overall': float(np.mean([m['pred_min_distance_overall'] for m in all_collab_metrics])),
    #         'per_sample_metrics': [
    #             {
    #                 'gt_mean_min_distance': float(m['gt_mean_min_distance']),
    #                 'gt_min_distance_overall': float(m['gt_min_distance_overall']),
    #                 'pred_mean_min_distance': float(m['pred_mean_min_distance']),
    #                 'pred_min_distance_overall': float(m['pred_min_distance_overall']),
    #             }
    #             for m in all_collab_metrics
    #         ]
    #     }
    #     with open(collab_metrics_path, 'w') as f:
    #         json.dump(collab_metrics_summary, f, indent=4)
        
    # Save the configuration (args)
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    print(f"Metrics and configuration successfully saved to {out_path}")

if __name__ == "__main__":
    main()

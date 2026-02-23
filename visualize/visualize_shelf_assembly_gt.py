import os
import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import smplx
import shutil
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil
# Use plot_3d_motion from humanml.utils but likely need to adapt it or use a custom one for 53 joints
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
from visualize.vis_utils import npy2obj # If needed, but we do it manually

def get_smplx_chain():
    # Standard SMPL-X kinematic chain (indices based on full 127 model or simplified)
    # We will assume we use the first 55 joints or similar.
    # We'll dynamically check output shape.
    
    # Body (22 joints: 0-21)
    # 0: Pelvis
    # 1: L_Hip, 2: R_Hip
    # 3: Spine1
    # 4: L_Knee, 5: R_Knee
    # 6: Spine2
    # 7: L_Ankle, 8: R_Ankle
    # 9: Spine3
    # 10: L_Foot, 11: R_Foot
    # 12: Neck
    # 13: L_Collar, 14: R_Collar
    # 15: Head
    # 16: L_Shoulder, 17: R_Shoulder
    # 18: L_Elbow, 19: R_Elbow
    # 20: L_Wrist, 21: R_Wrist
    
    # Hand Indices in SMPL-X (usually)
    # Left Hand (15 joints): 25..39
    # Right Hand (15 joints): 40..54
    # Jaw: 22, Eye_L: 23, Eye_R: 24
    
    # Kinematic Chains for plotting
    # [Root, Hip, Knee, Ankle, Foot]
    chain_r_leg = [0, 2, 5, 8, 11]
    chain_l_leg = [0, 1, 4, 7, 10]
    # [Root, Spine1, Spine2, Spine3, Neck, Head]
    chain_spine = [0, 3, 6, 9, 12, 15]
    # [Spine3, Collar, Shoulder, Elbow, Wrist]
    chain_r_arm = [9, 14, 17, 19, 21]
    chain_l_arm = [9, 13, 16, 18, 20]
    
    # Hands - Fingers
    # Left Hand (Wrist 20)
    # Thumb: 20 -> 37 -> 38 -> 39  (Indices need verification, standard smplx usually)
    # Index: 20 -> 25 -> 26 -> 27
    # Middle: 20 -> 28 -> 29 -> 30
    # Ring: 20 -> 31 -> 32 -> 33
    # Pinky: 20 -> 34 -> 35 -> 36
    
    # Fingers (start_idx, +1, +2)
    # L: 25, 28, 31, 34, 37
    # R: 40, 43, 46, 49, 52
    
    # Left Fingers
    chain_l_index = [20, 25, 26, 27]
    chain_l_middle = [20, 28, 29, 30]
    chain_l_ring = [20, 31, 32, 33]
    chain_l_pinky = [20, 34, 35, 36]
    chain_l_thumb = [20, 37, 38, 39] # Thumb usually last in 15-joint block
    
    # Right Hand (Wrist 21)
    chain_r_index = [21, 40, 41, 42]
    chain_r_middle = [21, 43, 44, 45]
    chain_r_ring = [21, 46, 47, 48]
    chain_r_pinky = [21, 49, 50, 51]
    chain_r_thumb = [21, 52, 53, 54]
    
    skeleton = [
        chain_r_leg, chain_l_leg, chain_spine, 
        chain_r_arm, chain_l_arm,
        chain_l_index, chain_l_middle, chain_l_ring, chain_l_pinky, chain_l_thumb,
        chain_r_index, chain_r_middle, chain_r_ring, chain_r_pinky, chain_r_thumb
    ]
    return skeleton

def main():
    print("Starting script...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1) # Visualize 1 by 1 or small batch
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--dataset", type=str, default="shelf_assembly")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    print(f"Loading dataset {args.dataset}...")
    # hml_mode='action' required for shelf_assembly to load all parts
    loader = get_dataset_loader(args.dataset, args.batch_size, num_frames=args.num_frames, split='train', hml_mode='action')
    
    iterator = iter(loader)
    try:
        motion, cond = next(iterator)
    except StopIteration:
        print("Dataset is empty.")
        return

    # motion shape: [batch_size, njoints, nfeats, frames]
    # In shelf_assembly: 
    # nfeats=6 (Rot6D)
    # njoints depends on concat in collate
    print(f"GT motion shape: {motion.shape}")
    
    B, J, D, T = motion.shape
    
    # Unpack Data
    # Permute to [B, T, J, D]
    motion_perm = motion.permute(0, 3, 1, 2).to(device) # [B, T, J, 6]
    
    # Initialize SMPL-X with correct batch size to avoid broadcasting errors
    # especially for lmk_bary_coords or other internal buffers
    num_samples = B * T
    model_path = "./body_models/smplx/SMPLX_NEUTRAL.npz"
    print(f"Initializing SMPL-X from {model_path} with batch_size={num_samples}...")
    smplx_model = smplx.create(model_path=model_path, model_type='smplx', use_pca=False, batch_size=num_samples).to(device)

    # Reshape for conversion
    rot6d = motion_perm.reshape(B*T, J, 6)
    rotmat = rotation_6d_to_matrix(rot6d) # [B*T, J, 3, 3]
    aa = matrix_to_axis_angle(rotmat) # [B*T, J, 3]
    
    # Root Translation (J=0 in dataset)
    # dataset.py: root_pos is (T, 3). Collate pads to (T, 1, 6).
    # So motion[:,0,:,:] is (B, 6, T). motion_perm[:,:,0,:] is (B, T, 6)
    root_transl = motion_perm[:, :, 0, :3].reshape(B*T, 3) # [B*T, 3]
    
    # Indices in concatenated tensor (from inspect_data_and_model.py check)
    # 0: Root (used for transl)
    # 1: Global Orient
    # 2-22: Body (21)
    # 23-37: Right Hand (15)
    # 38-52: Left Hand (15)
    
    global_orient = aa[:, 1, :] # [B*T, 3]
    body_pose = aa[:, 2:23, :].reshape(B*T, -1) # [B*T, 21*3 = 63]
    right_hand_pose = aa[:, 23:38, :].reshape(B*T, -1) # [B*T, 15*3 = 45]
    left_hand_pose = aa[:, 38:53, :].reshape(B*T, -1) # [B*T, 15*3 = 45]
    
    # Create zero buffers for other parts to avoid broadcasting issues
    num_samples = B * T
    dtype = global_orient.dtype
    
    betas = torch.zeros([num_samples, 10], dtype=dtype, device=device)
    expression = torch.zeros([num_samples, 10], dtype=dtype, device=device)
    jaw_pose = torch.zeros([num_samples, 3], dtype=dtype, device=device)
    leye_pose = torch.zeros([num_samples, 3], dtype=dtype, device=device)
    reye_pose = torch.zeros([num_samples, 3], dtype=dtype, device=device)
    
    print("Running SMPL-X forward pass...")
    with torch.no_grad():
        output = smplx_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=root_transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True
        )
    
    # output.joints: [B*T, 127, 3] usually
    joints = output.joints.reshape(B, T, -1, 3) # [B, T, J_smplx, 3]
    joints_np = joints.cpu().numpy()
    
    print(f"SMPL-X output joints shape: {joints.shape}")
    
    # Visualization
    out_path = "./save/visualize_shelf_assembly_gt"
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    
    print(f"Saving GT visualizations to {out_path}...")
    
    fps = 30.0
    skeleton = get_smplx_chain()
    
    texts = cond['y']['text']
    lengths = cond['y']['lengths'].numpy()
    
    for i in range(args.batch_size):
        length = lengths[i] if i < len(lengths) else args.num_frames
        caption = texts[i] if i < len(texts) else "Unknown"
        
        # [T, J, 3]
        m = joints_np[i, :length] 
        # plot_3d_motion expects [T, J, 3]
        
        save_file = os.path.join(out_path, f"gt_sample_{i:02d}.mp4")
        print(f"Rendering {save_file} (Text: {caption})...")
        
        # We use a custom skeleton, passes to plot_3d_motion
        # Note: plot_3d_motion calculates limits based on data
        try:
            ani = plot_3d_motion(save_file, skeleton, m, dataset="smplx", title=caption, fps=fps)
            ani.duration = length / fps
            ani.write_videofile(save_file, fps=fps, threads=4, logger=None)
            ani.close()
        except Exception as e:
            print(f"Visualization failed for sample {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Done. Visualizations saved in {out_path}")

if __name__ == "__main__":
    main()

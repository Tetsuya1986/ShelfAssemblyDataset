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

def main(args=None):
    if args is None:
        args = generate_args()
        
    # generate_args() / apply_rules() may strip prediction parameters if it loaded `task=generation` from the checkpoint args.json
    # We must explicitly force prediction mode here to evaluate futures.
    args.task = 'prediction'
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
    elif args.dataset == 'shelf_assembly':
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
                              hml_mode='action',
                              task='prediction',
                              input_seconds=args.input_seconds,
                              prediction_seconds=args.prediction_seconds,
                              stride=args.stride,
                              split='test',
                              autoregressive=args.autoregressive)

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
            
            bs = input_motion.shape[0]
            assert bs == 1
            
            # Total lengths of each sequence
            lengths = model_kwargs['y']['lengths']

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
            # gt_motion is [bs, nj, nfeats, slen]. Extract the prediction window.
            # The label (gt) prediction window starts at `history_len`.
            gt_max_len = int(lengths.max().item())
            gt_pred_window = input_motion[..., history_len:gt_max_len] # [bs, nj, nfeats, slen]
            
            # Use model.rot2xyz to get the xyz joint positions
            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask_gt = None if rot2xyz_pose_rep == 'xyz' else torch.ones((bs, gt_pred_window.shape[-1]), dtype=torch.bool, device=dist_util.dev())
            
            gt_xyz = model.rot2xyz(x=gt_pred_window, mask=rot2xyz_mask_gt, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                   jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                   get_rotations_back=False) # [bs, nj, 3, slen]
                                   
            # gt_xyz: [bs, nj, 3, slen] -> [bs, slen, nj, 3]
            gt_xyz_np = gt_xyz.permute(0, 3, 1, 2).cpu().numpy()

            # Process predictions chronologically to save memory, one repetition at a time
            pred_xyz_list = []
            for rep_i in range(args.num_repetitions):
                pred_chunk = batch_predictions[rep_i] # [bs, nj, nfeats, slen]
                rot2xyz_mask_pred = None if rot2xyz_pose_rep == 'xyz' else torch.ones((bs, pred_chunk.shape[-1]), dtype=torch.bool, device=dist_util.dev())
                
                pred_xyz_chunk = model.rot2xyz(x=pred_chunk, mask=rot2xyz_mask_pred, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                       get_rotations_back=False) # [bs, nj, 3, slen]
                pred_xyz_list.append(pred_xyz_chunk)
                
            # [K, bs, nj, 3, slen]
            pred_xyz = torch.stack(pred_xyz_list, dim=0)

            # pred_xyz: [K, bs, nj, 3, slen] -> [bs, K, slen, nj, 3]
            pred_xyz_np = pred_xyz.permute(1, 0, 4, 2, 3).cpu().numpy()

            # Calculate metrics per sample in the batch
            for i in range(bs):
                seq_len = lengths[i]
                if seq_len <= history_len:
                    continue # No future to predict
                
                # We only evaluate on the predicted future segment
                pred_future_len = seq_len - history_len
                
                # gt_xyz_np is [bs, slen, nj, 3], but it already sliced off history_len
                gt_future = gt_xyz_np[i, :pred_future_len, :, :] # [T, nj, 3]
                
                # If autoregressive included prefix, then prediction array starts at 0 with history
                if args.autoregressive_include_prefix:
                    pred_future = pred_xyz_np[i, :, history_len:history_len + pred_future_len, :, :]
                else:
                    pred_future = pred_xyz_np[i, :, :pred_future_len, :, :]
                
                # Check shapes match. Autoregressive sampler may generate slightly more frames than requested due to iterations.
                actual_pred_len = min(pred_future.shape[1], gt_future.shape[0])
                gt_future = gt_future[:actual_pred_len, ...]
                pred_future = pred_future[:, :actual_pred_len, ...]

                # Extract exactly the 52 structural joints corresponding to the rotated inputs.
                # SMPL-X output order: 0-21 Body, 22-24 Face, 25-39 Left Hand, 40-54 Right Hand, 55+ Face Contours.
                # The 53 input features are: 1 trans + 52 rotations (22 body, 15 LH, 15 RH).
                core_joints_indices = list(range(22)) + list(range(25, 55))
                
                gt_future = gt_future[:, core_joints_indices, :]
                pred_future = pred_future[:, :, core_joints_indices, :]

                ade, fde, mpjpe, apd = calculate_metrics(gt_future, pred_future)
                all_ade.append(ade)
                all_fde.append(fde)
                all_mpjpe.append(mpjpe)
                all_apd.append(apd)

            evaluated_count += bs

    if len(all_ade) == 0:
        print("No valid sequences found for evaluation.")
        return

    # Aggregate metrics
    mean_ade = np.mean(all_ade)
    mean_fde = np.mean(all_fde)
    mean_mpjpe = np.mean(all_mpjpe)
    mean_apd = np.mean(all_apd)

    res_str = (
        f"\n{'='*40}\n"
        f"Prediction Evaluation Metrics (K={args.num_repetitions})\n"
        f"{'='*40}\n"
        f"ADE@{args.num_repetitions}: {mean_ade:.4f}\n"
        f"FDE@{args.num_repetitions}: {mean_fde:.4f}\n"
        f"MPJPE@{args.num_repetitions}: {mean_mpjpe:.4f}\n"
        f"APD@{args.num_repetitions}: {mean_apd:.4f}\n"
        f"{'='*40}\n"
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
        
    # Save the configuration (args)
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    print(f"Metrics and configuration successfully saved to {out_path}")

if __name__ == "__main__":
    main()

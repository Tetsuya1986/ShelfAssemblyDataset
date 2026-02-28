import os
import torch
import numpy as np
import clip
from tqdm import tqdm
import re

def encode_directory(input_dir, output_dir, model, preprocess, device, batch_size=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy') and not f.endswith('_exclude')])
    
    for filename in tqdm(files, desc=f"Encoding {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, filename)
        
        # Create a directory for this video
        video_dir_name = filename.replace('.npy', '')
        video_output_dir = os.path.join(output_dir, video_dir_name)
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        else:
            # Check if already processed (quick check: if some frames exist)
            # This is a bit loose but helps with resumes
            if len(os.listdir(video_output_dir)) > 0:
                continue
            
        try:
            # Load images (Frames, H, W, C)
            images_np = np.load(input_path, mmap_mode='r')
            num_frames = images_np.shape[0]
            
            for i in range(0, num_frames, batch_size):
                batch_indices = range(i, min(i + batch_size, num_frames))
                batch_images = images_np[batch_indices] # (B, H, W, C)
                
                # Preprocess
                batch_images_torch = torch.from_numpy(batch_images).permute(0, 3, 1, 2).to(device)
                
                # CLIP ViT-B/32 normalization
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, 3, 1, 1)
                
                batch_images_torch = batch_images_torch.float() / 255.0
                batch_images_torch = torch.nn.functional.interpolate(batch_images_torch, size=(224, 224), mode='bicubic', align_corners=False)
                batch_images_torch = (batch_images_torch - mean) / std
                
                with torch.no_grad():
                    features = model.encode_image(batch_images_torch)
                    features_np = features.cpu().numpy()
                    
                    # Save each frame separately
                    for j, idx in enumerate(batch_indices):
                        frame_filename = f"frame_{idx:06d}.npy"
                        np.save(os.path.join(video_output_dir, frame_filename), features_np[j])
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    base_data_root = "/data/utsubo0/users/narita/shelf_assembly/shelf_assembly_dataset"
    
    # Headcam
    input_headcam = os.path.join(base_data_root, "headcam_img_npy")
    output_headcam = os.path.join(base_data_root, "headcam_clip_features")
    encode_directory(input_headcam, output_headcam, model, preprocess, device)
    
    # Envcam
    input_envcam = os.path.join(base_data_root, "envcam_img_npy")
    output_envcam = os.path.join(base_data_root, "envcam_clip_features")
    encode_directory(input_envcam, output_envcam, model, preprocess, device)

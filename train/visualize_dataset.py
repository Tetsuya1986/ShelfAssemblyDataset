# This code is based on https://github.com/openai/guided-diffusion
"""
Visualize dataset motion
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import numpy as np

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),
                              hml_mode='action')

    for motion, cond in data:
        dict_data = {}
        dict_data['motion'] = motion.cpu().detach().numpy()
        dict_data['text'] = cond['y']['text']

        f_name = 'dataset.npy'
        f_path = os.path.join(args.save_dir, f_name)
        np.save(f_path, dict_data)
        break


if __name__ == "__main__":
    main()

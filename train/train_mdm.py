# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import random
import string
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def main():
    args = train_args()
    fixseed(args.seed)
    # If save_dir doesn't exist, look for existing dirs with the same prefix + random suffix
    # to support resume via the base name (e.g. save/shelf_202602180113 finds save/shelf_202602180113_x7k2)
    if not os.path.exists(args.save_dir):
        import glob
        import random
        import string

        base = args.save_dir.rstrip("/")
        existing = sorted(glob.glob(base + "_????"))  # match 4-char suffix
        if existing:
            args.save_dir = existing[-1]  # use the latest matching directory
            print(f"Resuming from existing directory: {args.save_dir}")
        else:
            # Always create new random ID for base path
            run_id = "".join(
                random.SystemRandom().choices(string.ascii_lowercase + string.digits, k=4)
            )
            args.save_dir = base + "_" + run_id
            print(f"Creating new directory: {args.save_dir}")

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),
                              hml_mode='action')

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    if args.pretrained_checkpoint != "":
        from utils.model_util import load_pretrained_model
        load_pretrained_model(model, args.pretrained_checkpoint)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()

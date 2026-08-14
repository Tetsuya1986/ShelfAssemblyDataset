#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J action
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
python -m train.train_mdm --dataset comad --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --save_dir save/20260814_action_prediction_bert_50steps_action_2.0_CoMaD_woPre --overwrite --train_platform_type WandBPlatform --batch_size 512 --save_interval 10000 --arch trans_dec  --text_encoder_type bert

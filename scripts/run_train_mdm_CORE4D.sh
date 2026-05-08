#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J action
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
python -m train.train_mdm --dataset core4d --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --save_dir save/20260508_action_prediction_bert_50steps_action_2.0_CORE4D --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/narita/work/ShelfAssembly/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 32 --save_interval 10000

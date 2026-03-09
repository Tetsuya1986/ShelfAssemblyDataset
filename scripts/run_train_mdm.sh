#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J envcam_headcam_action_task_taskcommon
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
python -m train.train_mdm --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --save_dir save/20260301_action_prediction_bert_50steps_action_task_taskcommon_envcam_headcam --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/umagami/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 512 --save_interval 10000 --hml_mode action_task_taskcommon --use_envcam --use_headcam --pre_load_features


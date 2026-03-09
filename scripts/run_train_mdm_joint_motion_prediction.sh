#! /bin/bash
#SBATCH -p dgx-a100-40g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J joint_motion_prediction_action_task_taskcommon_envcam_headcam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
python -m train.train_mdm --dataset shelf_assembly --task joint_motion_prediction  --save_dir save/20260301_joint_motion_prediction_bert_50steps_action_task_taskcommon_envcam_headcam --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/umagami/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 32 --save_interval 10000 --hml_mode action_task_taskcommon --use_envcam --use_headcam --pre_load_features

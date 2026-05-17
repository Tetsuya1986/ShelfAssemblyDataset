#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J joint_motion_prediction_action_task_taskcommon_envcam_headcam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J

#python -m train.train_mdm --dataset shelf_assembly --task collab_prediction  --save_dir save/20260509_collab_prediction_bert_50steps_action_1.0 --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/umagami/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 128 --save_interval 10000 --label_option action
#python -m train.train_mdm --dataset shelf_assembly --task collab_prediction  --save_dir save/20260509_collab_prediction_bert_50steps_action_task_taskcommon_1.0 --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/umagami/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 128 --save_interval 10000 --label_option action_task_taskcommon
python -m train.train_mdm --dataset shelf_assembly --task collab_prediction --save_dir save/20260509_collab_prediction_bert_50steps_action_envcam_head_cam_1.0 --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/umagami/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 32 --save_interval 10000 --label_option action --use_envcam --use_headcam --pre_load_features

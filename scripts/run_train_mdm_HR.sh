#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J envcam_headcam_action_task_taskcommon
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
# python -m train.train_mdm --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/narita/work/ShelfAssembly/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 128 --save_interval 10000 --label_option action_taskcommon --data_sel HR-predictR --overwrite --save_dir save/20260816_action_prediction_bert_50steps_1.0_HR-predictR
# python -m train.train_mdm --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.5 --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/narita/work/ShelfAssembly/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 128 --save_interval 10000 --label_option action_taskcommon --data_sel HR-predictR --overwrite --save_dir save/20260816_action_prediction_bert_50steps_1.5_HR-predictR
python -m train.train_mdm --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --train_platform_type WandBPlatform --pretrained_checkpoint /home/mil/narita/work/ShelfAssembly/ShelfAssemblyDataset/save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 128 --save_interval 10000 --label_option action_taskcommon --data_sel HR-predictR --overwrite --save_dir save/20260816_action_prediction_bert_50steps_2.0_HR-predictR

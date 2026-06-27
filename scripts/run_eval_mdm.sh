#! /bin/bash
#SBATCH -p dgx-a100-40g
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -J joint_motion_prediction_action
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
python -m sample.eval_prediction --dataset shelf_assembly --task collab_prediction --input_seconds 0.5 --prediction_seconds 1.0 --label_option action --model_path save/20260626_collab_prediction_bert_50steps_action_envcam_1.0_1dzg/model000170000.pt  --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix --use_envcam


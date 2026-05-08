#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J action
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
python -m sample.eval_prediction --dataset core4d --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --model_path save/20260508_action_prediction_bert_50steps_action_2.0_CORE4D_z4ci/model000330000.pt  --split test --num_repetitions 1 --autoregressive --autoregressive_include_prefix --data_sel HH

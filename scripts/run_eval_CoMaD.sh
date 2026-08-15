#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 3-0
#SBATCH -J action
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
# python -m sample.eval_prediction --dataset comad --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --model_path save/20260814_action_prediction_bert_50steps_action_1.0_CoMaD_woPre_hp5y/model000250000.pt  --split test --num_repetitions 5 --autoregressive --autoregressive_include_prefix --data_sel HH
python -m sample.eval_prediction --dataset comad --task prediction --input_seconds 0.5 --prediction_seconds 1.5 --model_path save/20260814_action_prediction_bert_50steps_action_1.5_CoMaD_woPre_cjq2/model000240000.pt  --split test --num_repetitions 5 --autoregressive --autoregressive_include_prefix --data_sel HH
# python -m sample.eval_prediction --dataset comad --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --model_path save/20260814_action_prediction_bert_50steps_action_2.0_CoMaD_woPre_y1dd/model000600544.pt  --split test --num_repetitions 5 --autoregressive --autoregressive_include_prefix --data_sel HH

#! /bin/bash
#SBATCH -p dgx-a100-80g
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -J joint_motion_prediction_action
#SBATCH --mail-type=ALL
#SBATCH --mail-user=narita@mi.t.u-tokyo.ac.jp
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J
# python -m sample.eval_prediction --dataset shelf_assembly --task joint_motion_prediction --input_seconds 0.5 --prediction_seconds 1.0 --hml_mode action --model_path save/20260301_joint_motion_prediction_bert_50steps_action_18ld/model000240000.pt  --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix
# python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --label_option action --model_path save/20260223_action_prediction_bert_50steps_t7s2/model000600161.pt  --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix
# python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.5 --label_option action --model_path  save/20260228_action_prediction_bert_50steps_action_1.5_mxd6/model000600161.pt  --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix
# python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --label_option action --model_path  save/20260228_action_prediction_bert_50steps_action_2.0_qlxa/model000270000.pt  --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix
# python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --label_option action_task --model_path save/20260228_action_prediction_bert_50steps_action_task_mdyf/model000600161.pt  --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix
# python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.5 --label_option action_task --model_path save/20260228_action_prediction_bert_50steps_action_task_1.5_hjv3/model000600161.pt --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix
python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 2.0 --label_option action_task --model_path save/20260228_action_prediction_bert_50steps_action_task_2.0_fhqb/model000480000.pt --split test --num_repetitions 10 --autoregressive --autoregressive_include_prefix


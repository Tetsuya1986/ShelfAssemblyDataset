## Human Motion Prediction
* Train: `python -m train.train_mdm --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --save_dir save/{保存したい場所} --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint save/humanml_trans_dec_512_bert/model000600000.pt --batch_size 512 --save_interval 10000`みたいな感じで実行する。pretrained_checkpointと入出力層以外はモデル構造は同じで重みもloadした状態から学習を始める。
* Test: `python -m sample.eval_prediction --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --model_path save/shelf_pred_text-cond_from_humanml_trans_dec_512_bert_202602210336_bj1q/model000220000.pt --split test --num_repetitions 20 --autoregressive --autoregressive_include_prefix`みたいな感じで実行する。`--num_eval_samples 10`などをつけると、10データだけでテストできる。
* Generate: `python -m sample.generate --dataset shelf_assembly --task prediction --input_seconds 0.5 --prediction_seconds 1.0 --model_path save/shelf_pred_text-cond_from_humanml_trans_dec_512_bert_202602210336_bj1q/model000220000.pt --num_samples 2 --num_repetitions 1 --autoregressive --autoregressive_include_prefix`みたいな感じで動画などを作れる。


## Text to Motion Generation
* Train: `python -m train.train_mdm --dataset shelf_assembly --save_dir save/shelf_from_humanml_trans_dec_512_bert_202602191858 --overwrite --train_platform_type WandBPlatform --pretrained_checkpoint save/humanml_trans_dec_512_bert/model000600000.pt --save_interval 25000`みたいな感じで実行する。pretrained_checkpointと入出力層以外はモデル構造は同じで重みもloadした状態から学習を始める。
* Generation: `python -m sample.generate --model_path save/shelf_from_humanml_trans_dec_512_bert_202602191858_7e6a/model000100000.pt --num_samples 10 --num_repetitions 3`など



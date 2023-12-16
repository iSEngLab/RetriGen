# NewDataSet
parent_dir="../../RetriGen/hybrid/result/NewDataSet"
model_name="new_alpha_0.5.bin"
output_name="new_alpha_0.5"
train_data_file="${parent_dir}/train/assert_train_new_alpha_0.5.csv"
test_data_file="${parent_dir}/test/assert_test_new_alpha_0.5.csv"
val_data_file="${parent_dir}/val/assert_val_new_alpha_0.5.csv"

# OldDataSet
#model_name="old_alpha_0.5.bin"
#output_name="old_alpha_0.5"
#parent_dir="../../RetriGen/hybrid/result/OldDataSet"
#train_data_file="${parent_dir}/train/assert_train_old_alpha_0.5.csv"
#test_data_file="${parent_dir}/test/assert_test_old_alpha_0.5.csv"
#val_data_file="${parent_dir}/val/assert_val_old_alpha_0.5.csv"

# train
python graphcodebert_main.py  \
  --output_dir=./saved_models \
  --model_name=${model_name} \
  --do_train \
  --train_data_file=${train_data_file} \
  --eval_data_file=${val_data_file} \
  --test_data_file=${test_data_file} \
  --epochs 75 \
  --encoder_block_size 512 \
  --decoder_block_size 256 \
  --train_batch_size 12 \
  --eval_batch_size 12 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --n_gpu 2 \
  --evaluate_during_training \
  --seed 123456  2>&1 | tee train.log

# test
python graphcodebert_main.py \
  --output_name ${output_name} \
  --output_dir=./saved_models \
  --model_name=${model_name} \
  --do_test \
  --test_data_file=${test_data_file} \
  --encoder_block_size 512 \
  --decoder_block_size 256 \
  --beam_size 1 \
  --eval_batch_size 1 \
  --n_gpu 1
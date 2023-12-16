output_dir="./result/"
output_model_name="new_alpha_0.5.bin"
parent_dir="../../RetriGen/hybrid/result/NewDataSet"
train_filename="${parent_dir}/train/assert_train_new_alpha_0.5.csv"
test_filename="${parent_dir}/test/assert_test_new_alpha_0.5.csv"
dev_filename="${parent_dir}/val/assert_val_new_alpha_0.5.csv"

# OldDataSet
#output_model_name="old_alpha_0.5.bin"
#parent_dir="../../RetriGen/hybrid/result/OldDataSet"
#train_filename="${parent_dir}/train/assert_train_old_alpha_0.5.csv"
#test_filename="${parent_dir}/test/assert_test_old_alpha_0.5.csv"
#dev_filename="${parent_dir}/val/assert_val_old_alpha_0.5.csv"

python run.py \
  --do_train \
  --do_eval \
  --model_type gpt2 \
  --output_model_name ${output_model_name} \
  --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
  --train_filename ${train_filename} \
  --dev_filename ${dev_filename}  \
  --test_filename ${test_filename}  \
  --output_dir ${output_dir} \
  --max_source_length 512 \
  --max_target_length 512 \
  --beam_size 1 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 2>&1 | tee ${output_dir}/train.log

python run.py \
  --do_test \
  --model_type gpt2 \
  --output_file_name ${output_model_name} \
  --load_model_path ${output_dir}/checkpoint-best-ppl/new_alpha_0.5.bin \
  --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
  --test_filename ${test_filename}  \
  --output_dir ${output_dir} \
  --max_source_length 512 \
  --max_target_length 512 \
  --beam_size 1 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 2>&1 | tee ${output_dir}/test.log
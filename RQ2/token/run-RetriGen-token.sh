train_data_file="./assert_train_new_token.csv"
test_data_file="./assert_test_new_token.csv"
val_data_file="./assert_val_new_token.csv"

train_batch_size=8
eval_batch_size=8
test_batch_size=4

result_file_path="./RetriGen_new_token.txt"
pred_file_path="./RetriGen_new_token_prediction.csv"

python codet5_main.py \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --do_eval \
    --train_data_file=${train_data_file} \
    --test_data_file=${test_data_file} \
    --eval_data_file=${val_data_file} \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --test_batch_size ${test_batch_size} \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --n_gpu 2 \
    --evaluate_during_training \
    --num_beams 1 \
    --result_file_path ${result_file_path} \
    --pred_file_path ${pred_file_path} \
    --seed 123456

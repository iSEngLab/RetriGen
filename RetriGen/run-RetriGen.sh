train_data_file="datas/train.jsonl"
test_data_file="datas/test.jsonl"
val_data_file="datas/valid.jsonl"

train_batch_size=16
eval_batch_size=16
test_batch_size=16

result_file_path="result/NewDataSet/RetriGen_new_alpha.txt"
pred_file_path="result/NewDataSet/RetriGen_new_alpha_prediction.csv"

CUDA_VISIBLE_DEVICES=1 python codet5_main.py \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --do_eval \
    --train_data_file=${train_data_file} \
    --test_data_file=${test_data_file} \
    --eval_data_file=${val_data_file} \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size  \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --test_batch_size ${test_batch_size} \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --n_gpu 1 \
    --evaluate_during_training \
    --num_beams 1 \
    --result_file_path ${result_file_path} \
    --pred_file_path ${pred_file_path} \
    --seed 123456

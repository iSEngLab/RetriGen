test_data_file="../../RetriGen/hybrid/result/NewDataSet/test/assert_test_new_alpha_0.5.csv"
result_file_path="./result/RetriGen_new_none.txt"
pred_file_path="./result/RetriGen_new_none_prediction.csv"

CUDA_VISIBLE_DEVICES=0 python codet5_main.py \
    --do_test \
    --test_data_file=${test_data_file} \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --num_beams 1 \
    --test_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --n_gpu 1 \
    --result_file_path ${result_file_path}  \
    --pred_file_path ${pred_file_path} \
    --seed 123456
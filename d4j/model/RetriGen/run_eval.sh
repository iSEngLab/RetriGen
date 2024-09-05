input_file=$1
base_dir=`dirname ${input_file}`

echo ${base_dir}

CUDA_VISIBLE_DEVICES=0 python codet5_main.py \
    --output_dir={base_dir}/Joint_preds \
    --do_test \
    --test_data_file=${input_file} \
    --encoder_block_size 512 \
    --decoder_block_size 256  \
    --test_batch_size 8 \
    --n_gpu 1 \
    --num_beams 1 \
    --result_file_path ${result_file_path} \
    --pred_file_path ${pred_file_path} \
    --seed 123456

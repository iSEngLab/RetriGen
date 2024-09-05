CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python llama_sft.py \
            --do_train \
            --do_eval \
            --bf16 \
            --model_name_or_path /data/Models/CodeLLama/base/codellama-7b-hf/ \
            --model_max_length 1024 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 8 \
            --auto_find_batch_size \
            --gradient_accumulation_steps 1 \
            --dataloader_num_workers 8 \
            --dataloader_pin_memory \
            --datalodaer_prefetch_factor 4 \
            --gradient_accumulation_steps 1 \
            --data_path ${DATA_PATH} \
            --train_filename train.jsonl \
            --eval_filename valid.jsonl \
            --test_filename test.jsonl \
            --learning_rate 5e-5 \
            --eval_strategy epoch \
            --save_strategy epoch \
            --greater_is_better False \
            --logging_steps 10 \
            --num_train_epochs 5 \
            --save_total_limit 3 \
            --output_dir ${OUTPUT_DIR} \
            --report_to tensorboard \
            --early_stop_patience 2 \
            --plot_loss

CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3

PYTHONUNBUFFERED=1 DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} accelerate launch llama_sft.py \
            --do_train \
            --do_eval \
            --fp16 \
            --model_name_or_path ./codellama-7b-hf \
            --model_max_length 1024 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --dataloader_num_workers 2 \
            --dataloader_pin_memory \
            --datalodaer_prefetch_factor 2 \
            --gradient_accumulation_steps 1 \
            --data_path ${DATA_PATH} \
            --train_filename train.jsonl \
            --eval_filename valid.jsonl \
            --test_filename test.jsonl \
            --learning_rate 5e-5 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --greater_is_better False \
            --logging_steps 50 \
            --num_train_epochs 3 \
            --save_total_limit 3 \
            --output_dir ${OUTPUT_DIR} \
            --report_to tensorboard \
            --plot_loss
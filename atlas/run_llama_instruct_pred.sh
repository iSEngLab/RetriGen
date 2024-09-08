CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python llama_pred.py \
            --model_name_or_path ${OUTPUT_DIR} \
            --model_max_length 1024 \
            --output_dir ${OUTPUT_DIR} \
            --use_instruct \
            --max_new_tokens 256 \
            --max_length 1024 \
            --num_beams 10 \
            --num_retrun_sequences 1

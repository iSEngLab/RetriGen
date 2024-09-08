TOTAL=6

for i in `seq 2 ${TOTAL}`;do
  python hybrid_retrieve.py \
      --codebase_data_path data/train_embedding.jsonl \
      --query_data_path data/evosuite_buggy_tests/${i}/input_embedding.jsonl \
      --output_dir ./results/${i} \
      --output_filename inputs_retrieval_results.jsonl \
      --batch_size 100 \
      --cpu_count 65 2>&1 | tee logs/${i}.log
done
if [ $? -ne 0 ]; then
    echo "Retrieval Failed!"
    exit 0
fi
# dataset file path
codebase_path="../dataset/NewDataSet/assert_train_new_embedding.csv"
query_data_path="../dataset/NewDataSet/assert_val_new_embedding.csv"

#codebase_path="../dataset/OldDataSet/assert_train_old_embedding.csv"
#query_data_path="../dataset/OldDataSet/assert_val_old_embedding.csv"

# result file path
output_sim_path="./result/NewDataSet/codebase_train_query_val/total_sim_{}.parquet"
output_result_path="./result/NewDataSet/codebase_train_query_val/total_result.csv"

#output_sim_path="./result/OldDataSet/codebase_train_query_val/total_sim_{}.parquet"
#output_result_path="./result/OldDataSet/codebase_train_query_val/total_result.csv"

# hyperparameters
# calc batch size in one subprocess
batch_size=1000
# similarity storage size
storage_size=1000
# subprocess count
cpu_count=20

python retrieval_by_codellama.py \
  --codebase_path ${codebase_path} \
  --query_data_path ${query_data_path}  \
  --batch_size ${batch_size} \
  --storage_size ${storage_size}  \
  --output_sim_path ${output_sim_path} \
  --output_result_path ${output_result_path}  \
  --cpu_count ${cpu_count}

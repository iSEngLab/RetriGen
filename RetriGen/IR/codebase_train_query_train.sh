# dataset file path
codebase_path="../dataset/NewDataSet/assert_train_new.csv"
query_data_path="../dataset/NewDataSet/assert_train_new.csv"

#codebase_path="../dataset/OldDataSet/assert_train_old.csv"
#query_data_path="../dataset/OldDataSet/assert_train_old.csv"

# result file path
output_sim_path="./result/NewDataSet/codebase_train_query_train/total_sim_{}.parquet"
output_result_path="./result/NewDataSet/codebase_train_query_train/total_result.csv"

#output_sim_path="./result/OldDataSet/codebase_train_query_train/total_sim_{}.parquet"
#output_result_path="./result/OldDataSet/codebase_train_query_train/total_result.csv"

# hyperparameters
# calc batch size in one subprocess
batch_size=200
# similarity storage size
storage_size=2000
# subprocess count
cpu_count=60

python retrieval_by_IR.py \
  --codebase_path ${codebase_path} \
  --query_data_path ${query_data_path}  \
  --batch_size ${batch_size} \
  --storage_size ${storage_size}  \
  --output_sim_path ${output_sim_path}  \
  --output_result_path ${output_result_path}  \
  --cpu_count ${cpu_count}

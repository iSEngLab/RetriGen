codellama_sim_path="../codellama/result/NewDataSet/codebase_train_query_train/"
IR_sim_path="../IR/result/NewDataSet/codebase_train_query_train/"

alpha=0.5

output_result_path="./result/NewDataSet/train/alpha_0.5_total_result.csv"
output_dataset_path="./result/NewDataSet/train/assert_train_new_alpha_0.5.csv"
query_dataset="../dataset/NewDataSet/assert_train_new.csv"
codebase_dataset="../dataset/NewDataSet/assert_train_new.csv"
cpu_count=24

python retrieval_by_hybrid.py  \
  --codellama_sim_path ${codellama_sim_path}  \
  --IR_sim_path ${IR_sim_path} \
  --save_result \
  --save_dataset  \
  --alpha ${alpha}  \
  --output_result_path ${output_result_path}  \
  --output_dataset_path ${output_dataset_path} \
  --query_dataset ${query_dataset}  \
  --codebase_dataset ${codebase_dataset}  \
  --cpu_count ${cpu_count}
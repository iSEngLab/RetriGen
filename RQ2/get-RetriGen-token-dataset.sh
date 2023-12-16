# get train set
train_result_file="../RetriGen/IR/result/NewDataSet/codebase_train_query_train/total_result.csv"
train_output_file="./token/assert_train_new_token.csv"

python dataset.py \
  --result_file ${train_result_file}  \
  --output_file ${train_output_file}

# get test set
test_result_file="../RetriGen/IR/result/NewDataSet/codebase_train_query_test/total_result.csv"
test_output_file="./token/assert_test_new_token.csv"

python dataset.py \
  --result_file ${test_result_file}  \
  --output_file ${test_output_file}

# get val set
val_result_file="../RetriGen/IR/result/NewDataSet/codebase_train_query_val/total_result.csv"
val_output_file="./token/assert_val_new_token.csv"

python dataset.py \
  --result_file ${val_result_file}  \
  --output_file ${val_output_file}
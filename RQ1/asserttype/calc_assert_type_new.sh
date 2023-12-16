data_file="../../RetriGen/hybrid/Result/NewDataSet/test/assert_test_new_alpha_0.5.csv"
result_file="../../RetriGen/result/NewDataSet/CodeT5_new_alpha.txt"
output_file="./assertion_type_new.csv"

python calclate.py  \
  --data_file ${data_file}  \
  --result_file ${result_file}  \
  --output_file ${output_file}
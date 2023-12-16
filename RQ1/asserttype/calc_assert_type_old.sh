data_file="../../RetriGen/hybrid/Result/OldDataSet/test/assert_test_old_alpha_0.5.csv"
result_file="../../RetriGen/result/OldDataSet/CodeT5_oldalpha.txt"
output_file="./assertion_type_old.csv"

python calclate.py  \
  --data_file ${data_file}  \
  --result_file ${result_file}  \
  --output_file ${output_file}
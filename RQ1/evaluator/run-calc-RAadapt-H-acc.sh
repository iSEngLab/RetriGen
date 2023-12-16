# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/IR_Label.txt \
  --result_file ../result/NewDataSet/RAadapt-H_Result.txt)

echo "RAadapt-H NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/IR_Label.txt \
  --result_file ../result/OldDataSet/RAadapt-H_Result.txt)

echo "RAadapt-H OldDataSet ${old_result}"
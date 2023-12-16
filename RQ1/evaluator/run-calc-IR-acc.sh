# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/IR_Label.txt \
  --result_file ../result/NewDataSet/IR_Result.txt)

echo "IR NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/IR_Label.txt \
  --result_file ../result/OldDataSet/IR_Result.txt)

echo "IR OldDataSet ${old_result}"
# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/IR_Label.txt \
  --result_file ../result/NewDataSet/RAadapt-NN_Result.txt)

echo "RAadapt-NN NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/IR_Label.txt \
  --result_file ../result/OldDataSet/RAadapt-NN_Result.txt)

echo "RAadapt-NN OldDataSet ${old_result}"
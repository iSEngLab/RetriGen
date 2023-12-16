# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/Integration_Label.txt \
  --result_file ../result/NewDataSet/Integration_Result.txt)

echo "Integration NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/Integration_Label.txt \
  --result_file ../result/OldDataSet/Integration_Result.txt)

echo "Integration OldDataSet ${old_result}"
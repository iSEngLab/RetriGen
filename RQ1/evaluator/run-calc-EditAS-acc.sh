# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/EditAS_Label.txt \
  --result_file ../result/NewDataSet/EditAs_Result.txt)

echo "EditAS NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/EditAS_Label.txt \
  --result_file ../result/OldDataSet/EditAs_Result.txt)

echo "EditAS OldDataSet ${old_result}"
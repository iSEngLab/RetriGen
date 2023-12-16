# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/RetriGen_Label.txt \
  --result_file ../result/NewDataSet/RetriGen_Result.txt)

echo "RetriGen NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/RetriGen_Label.txt \
  --result_file ../result/OldDataSet/RetriGen_Result.txt)

echo "RetriGen OldDataSet ${old_result}"
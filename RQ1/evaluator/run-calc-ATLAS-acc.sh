# NewDataSet accuracy
new_result=$(python acc.py \
  --label_file ../result/NewDataSet/ATLAS_Label.txt \
  --result_file ../result/NewDataSet/ATLAS_Result.txt)

echo "ATLAS NewDataSet ${new_result}"

# OldDataSet accuracy
old_result=$(python acc.py \
  --label_file ../result/OldDataSet/ATLAS_Label.txt \
  --result_file ../result/OldDataSet/ATLAS_Result.txt)

echo "ATLAS OldDataSet ${old_result}"
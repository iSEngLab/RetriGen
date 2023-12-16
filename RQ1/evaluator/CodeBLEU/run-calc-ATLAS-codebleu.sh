# NewDataSet CodeBLEU
new_result=$(python calc_code_bleu.py \
  --refs ../../result/NewDataSet/ATLAS_Label.txt \
  --hyp ../../result/NewDataSet/ATLAS_Result.txt \
  --lang java)

echo "ATLAS NewDataSet ${new_result}"

# OldDataSet CodeBLEU
old_result=$(python calc_code_bleu.py \
  --refs ../../result/OldDataSet/ATLAS_Label.txt \
  --hyp ../../result/OldDataSet/ATLAS_Result.txt \
  --lang java)

echo "ATLAS OldDataSet ${old_result}"
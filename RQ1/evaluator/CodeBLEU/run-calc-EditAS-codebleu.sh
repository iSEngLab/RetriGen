# NewDataSet CodeBLEU
new_result=$(python calc_code_bleu.py \
  --refs ../../result/NewDataSet/EditAS_Label.txt \
  --hyp ../../result/NewDataSet/EditAS_Result.txt \
  --lang java)

echo "EditAS NewDataSet ${new_result}"

# OldDataSet CodeBLEU
old_result=$(python calc_code_bleu.py \
  --refs ../../result/OldDataSet/EditAS_Label.txt \
  --hyp ../../result/OldDataSet/EditAS_Result.txt \
  --lang java)

echo "EditAS OldDataSet ${old_result}"
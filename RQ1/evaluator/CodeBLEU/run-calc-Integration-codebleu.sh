# NewDataSet CodeBLEU
new_result=$(python calc_code_bleu.py \
  --refs ../../result/NewDataSet/Integration_Label.txt \
  --hyp ../../result/NewDataSet/Integration_Result.txt \
  --lang java)

echo "Integration NewDataSet ${new_result}"

# OldDataSet CodeBLEU
old_result=$(python calc_code_bleu.py \
  --refs ../../result/OldDataSet/Integration_Label.txt \
  --hyp ../../result/OldDataSet/Integration_Result.txt \
  --lang java)

echo "Integration OldDataSet ${old_result}"
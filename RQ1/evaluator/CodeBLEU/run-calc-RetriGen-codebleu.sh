# NewDataSet CodeBLEU
new_result=$(python calc_code_bleu.py \
  --refs ../../result/NewDataSet/RetriGen_Label.txt \
  --hyp ../../result/NewDataSet/RetriGen_Result.txt \
  --lang java)

echo "RetriGen NewDataSet ${new_result}"

# OldDataSet CodeBLEU
old_result=$(python calc_code_bleu.py \
  --refs ../../result/OldDataSet/RetriGen_Label.txt \
  --hyp ../../result/OldDataSet/RetriGen_Result.txt \
  --lang java)

echo "RetriGen OldDataSet ${old_result}"
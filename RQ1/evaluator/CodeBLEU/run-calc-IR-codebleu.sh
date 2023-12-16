# NewDataSet CodeBLEU
new_result=$(python calc_code_bleu.py \
  --refs ../../result/NewDataSet/IR_Label.txt \
  --hyp ../../result/NewDataSet/IR_Result.txt \
  --lang java)

echo "IR NewDataSet ${new_result}"

# OldDataSet CodeBLEU
old_result=$(python calc_code_bleu.py \
  --refs ../../result/OldDataSet/IR_Label.txt \
  --hyp ../../result/OldDataSet/IR_Result.txt \
  --lang java)

echo "IR OldDataSet ${old_result}"
# NewDataSet CodeBLEU
new_result=$(python calc_code_bleu.py \
  --refs ../../result/NewDataSet/IR_Label.txt \
  --hyp ../../result/NewDataSet/RAadapt-NN_Result.txt \
  --lang java)

echo "RAadapt-NN NewDataSet ${new_result}"

# OldDataSet CodeBLEU
old_result=$(python calc_code_bleu.py \
  --refs ../../result/OldDataSet/IR_Label.txt \
  --hyp ../../result/OldDataSet/RAadapt-NN_Result.txt \
  --lang java)

echo "RAadapt-NN OldDataSet ${old_result}"
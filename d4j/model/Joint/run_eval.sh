input_file=$1
base_dir=`dirname ${input_file}`

echo ${base_dir}

python model/Joint/run.py \
	--do_test \
	--model_name_or_path ../codet5-base \
	--train_filename ../dataset/NewDataSet/assert_train_new.csv \
	--test_filename ${input_file} \
	--output_dir ${base_dir}/Joint_preds \
	--max_source_length 512 \
	--max_target_length 64 \
	--code_length 256 \
	--nl_length 64 \
	--beam_size 10 \
	--test_batch_size 4 \
	--GPU_ids 0,1
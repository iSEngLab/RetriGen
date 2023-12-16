# NewDataSet
parent_dir="../dataset/NewDataSet"
source_paths=("${parent_dir}/assert_train_new.csv" "${parent_dir}/assert_test_new.csv" "${parent_dir}/assert_val_new.csv")
target_paths=("${parent_dir}/assert_train_new_embedding.csv" "${parent_dir}/assert_test_new_embedding.csv" "${parent_dir}/assert_val_new_embedding.csv")

# OldDataSet
#parent_dir="../dataset/OldDataSet"
#source_paths=("${parent_dir}/assert_train_old.csv" "${parent_dir}/assert_test_old.csv" "${parent_dir}/assert_val_old.csv")
#target_paths=("${parent_dir}/assert_train_old_embedding.csv" "${parent_dir}/assert_test_old_embedding.csv" "${parent_dir}/assert_val_old_embedding.csv")

for i in $(seq 1 `expr ${#source_paths[@]}`); do
    source=${source_paths[i]}
    target=${target_paths[i]}

    python embed.sh \
        --source_path ${source} \
        --target_path ${target}
done
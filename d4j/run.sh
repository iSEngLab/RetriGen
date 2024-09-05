#!/bin/bash

N_JOBS=${1:-20}
TOTAL=10

BUGGY_GEN_DIR="data/evosuite_buggy_regression_all"
BUGGY_TEST_DIR="data/evosuite_buggy_tests"

for i in `seq 1 ${TOTAL}`;do
    python -m extractor.main gen_tests ${i} --out_dir ${BUGGY_GEN_DIR}/${i}/generated --suffix b --n_jobs ${NJOBS}
done
if [ $? -ne 0 ]; then
    echo "Generate Test Failed!"
    exit 0
fi

for i in `seq 1 ${TOTAL}`;do
    python -m extractor.main prepare_tests ${BUGGY_GEN_DIR}/${i}/generated
done
if [ $? -ne 0 ]; then
    echo "Prepare Tests Failed!"
    exit 0
fi

for i in `seq 1 ${TOTAL}`;do
    python -m extractor.main ex_tests ${BUGGY_GEN_DIR}/${i} --output_dir ${BUGGY_TEST_DIR}/${i}
done
if [ $? -ne 0 ]; then
    echo "Execute Test Failed!"
    exit 0
fi


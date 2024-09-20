N_JOBS=${1:-20}
TOTAL=1

BUGGY_GEN_DIR="data/evosuite_buggy_regression_all"
BUGGY_TEST_DIR="data/evosuite_buggy_tests"
MODEL_NAME=RetriGen

#for i in `seq 1 ${TOTAL}`;do
#    python toga.py ${BUGGY_TEST_DIR}/${i}/inputs.csv ${BUGGY_TEST_DIR}/${i}/meta.csv ${MODEL_NAME}
#done
#if [ $? -ne 0 ]; then
#    echo "Toga oracles generated Failed!"
#    exit 0
#fi
#
#for i in `seq 1 ${TOTAL}`;do
#    python naive.py ${BUGGY_TEST_DIR}/${i}/inputs.csv ${BUGGY_TEST_DIR}/${i}/meta.csv ${MODEL_NAME}
#done
#if [ $? -ne 0 ]; then
#    echo "Naive oracles generated Failed!"
#    exit 0
#fi

g=toga
START=${2:-1}
END=10

for i in `seq ${START} ${END}`;do
    echo ${i}
    for t in buggy;do
        echo ${t}
        echo ${g}
        bash run_exp.sh data/evosuite_${t}_tests/${i} data/evosuite_${t}_regression_all/${i} ${N_JOBS} ${g} ${MODEL_NAME}
    done
    rm -rf /tmp/run_bug_detection.pl_*
done

rm -rf /tmp/run_bug_detection.pl_*
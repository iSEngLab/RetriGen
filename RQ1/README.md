# RQ1

## How to calc accuracy

run the command below and get the accuracy result:

```bash
cd evaluator

sh run-calc-ATLAS-acc.sh
sh run-calc-EditAS-acc.sh
sh run-calc-Integration-acc.sh
sh run-calc-IR-acc.sh
sh run-calc-RAadapt-H-acc.sh
sh run-calc-RAadapt-NN-acc.sh
sh run-calc-RetriGen-acc.sh
```

## How to calc CodeBLEU

run the command below to get the `CodeBLEU result`:

```bash
cd evaluator/CodeBLEU

sh run-calc-ATLAS-codebleu.sh
sh run-calc-EditAS-codebleu.sh
sh run-calc-Integration-codebleu.sh
sh run-calc-IR-codebleu.sh
sh run-calc-RAadapt-H-codebleu.sh
sh run-calc-RAadapt-NN-codebleu.sh
sh run-calc-RetriGen-codebleu.sh
```

## How to calc assert type

run the command below to get the assert type results:

```bash
cd asserttype
sh calc_assert_type_new.sh
sh calc_assert_type_old.sh
```
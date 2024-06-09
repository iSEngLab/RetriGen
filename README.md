# Improving Deep Assertion Generation via Fine-Tuning Retrieval-Augmented Pre-trained Language Models

## Environment Setup

```bash
conda env create --name RetriGen python=3.9
conda activate RetriGen
pip install -r requirements.txt
```

tips: torch version may depend on CUDA version, so you should check the version of CUDA and replace it in `requirements.txt`.

## Folder Structure

```bash
├── RQ1
│   ├── asserttype : the source code to get assert type result
│   ├── evaluator : the source code for metrics
│   └── result : the result of RQ1
├── RQ2
│   ├── dataset.py : source code to get fine-tune dataset
│   ├── embed : source code to fine-tune and test RetriGen-embed model
│   ├── get-RetriGen-embed-dataset.sh : script to get fine-tune embed-based dataset
│   ├── get-RetriGen-token-dataset.sh : script to get fine-tune token-based dataset
│   ├── none : source code to test none fine-tune model
│   └── token : source code to fine-tune and test RetriGen-token model
├── RQ3
│   ├── CodeBERT : source code and result to fine-tune CodeBERT
│   ├── CodeGPT : source code and result to fine-tune CodeGPT
│   ├── GraphCodeBERT : source code and result to fine-tune GraphCodeBERT
│   └── UniXcoder : source code and result to fine-tune UniXcoder
├── RetriGen
│   ├── IR : source code to get token-based retrieval result
│   ├── codellama : source code to get embed-based retrieval result
│   ├── codet5_main.py
│   ├── dataset : raw dataset
│   ├── hybrid : source code to get hybrid retrieval result
│   └── run-RetriGen.sh : main script to fine-tune CodeT5 model and test to get RetriGen result
└── requirements.txt
```

## Fine-tuned model and data

Due to the huge model size, we are unable to upload trained models to the anonymous website. 
All trained models will be available upon accecption.



Raw data are in `./RetriGen/dataset` folder, following commands below to get dataset after Retrival

### Unzip dataset

```bash
cd RetriGen/dataset
unzip NewDataSet.zip
unzip OldDataSet.zip
```

### Get codellama embedding result

Using [Ollama](http://ollama.ai) to deploy `codellama:7b` to get `codellama` embedding results.

#### How to deploy Ollama

Check [here](https://ollama.ai/download) to download Ollama.

After download successfully, using commands below to deploy `codellama:7b`

```bash
ollama pull codellama:7b
ollama serve
```

#### Reproduce result

After deploy `codellama:7b` locally, run the command below:

```bash
cd RetriGen/codellama
sh embed.sh
```

### Get RetriGen dataset

run the commands below (taking NewDataSet as an example, OldDataSet needs to replace the dataset path in the script) to get RetriGen fine-tune dataset:

```bash
# using IR to get token-based retrieval result
cd RetriGen/IR
mkdir -p result/{NewDataSet,OldDataSet}/codebase_train_query_{train,test,val}
sh codebase_train_query_train.sh
sh codebase_train_query_test.sh
sh codebase_train_query_val.sh

# using codellama to get embedding-based retrieval result
cd RetriGen/codellama
mkdir -p result/{NewDataSet,OldDataSet}/codebase_train_query_{train,test,val}
sh codebase_train_query_train.sh
sh codebase_train_query_test.sh
sh codebase_train_query_val.sh

# using hybrid to get RetriGen dataset
cd RetriGen/hybrid
mkdir -p result/{NewDataSet,OldDataSet}/{train,test,val}
sh test_alpha_0.5.sh
sh train_alpha_0.5.sh
sh val_alpha_0.5.sh
```

## Fine-tune and test CodeT5 to get RetriGen result

run the commands below to fine-tune CodeT5 and test:

```bash
cd RetriGen
mkdir -p result/{NewDataSet,OldDataSet}
sh run-RetriGen.sh
```

## RQ1

### Calculate accuracy

run the commands below and get the accuracy result:

```bash
cd RQ1/evaluator

sh run-calc-ATLAS-acc.sh
sh run-calc-EditAS-acc.sh
sh run-calc-Integration-acc.sh
sh run-calc-IR-acc.sh
sh run-calc-RAadapt-H-acc.sh
sh run-calc-RAadapt-NN-acc.sh
sh run-calc-RetriGen-acc.sh
```

### Calculate CodeBLEU

run the commands below to get the `CodeBLEU result`:

```bash
cd RQ1/evaluator/CodeBLEU

sh run-calc-ATLAS-codebleu.sh
sh run-calc-EditAS-codebleu.sh
sh run-calc-Integration-codebleu.sh
sh run-calc-IR-codebleu.sh
sh run-calc-RAadapt-H-codebleu.sh
sh run-calc-RAadapt-NN-codebleu.sh
sh run-calc-RetriGen-codebleu.sh
```

### Calculate assert type

run the commands below to get the assert type results:

```bash
cd RQ1/asserttype
sh calc_assert_type_new.sh
sh calc_assert_type_old.sh
```

## RQ2

### Get RetriGen-token and RetriGen-embed dataset

run the commands below to get `RetiGen-token` and `RetriGen-embed` training dataset:

```bash
cd RQ2
sh get-RetriGen-token-dataset.sh
sh get-RetriGen-embed-dataset.sh
```

### Reproduce the result

```bash
cd RQ2
sh embed/run-RetriGen-embed.sh
sh none/run-RetriGen-none.sh
sh token/run-RetriGen-token.sh
```

after reproduce the result, you can use RQ1 code to calc accuracy and CodeBLEU

## RQ3

**The Appendix Table showing comparisons of selected PLMs in RQ3 is uploaded in our [Appendix.pdf](https://anonymous.4open.science/r/RetriGen/Appendix.pdf)**

run the following commands in the model you want to reproduce the result in RQ3:

```bash
cd /the model you want
sh ${run-xxx.sh} # xxx is the model name
```

you can use RQ1 code to calc accuracy and `CodeBLEU`


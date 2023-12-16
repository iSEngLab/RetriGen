# RetriGen

## How to get RetriGen result

1. using IR to get token-based retrieval result
2. using codellama to get embedding-based retrieval result
3. using hybrid to get RetriGen dataset
4. train and test CodeT5 model to get RetriGen result

## How to reproduce result

### unzip dataset

```bash
cd dataset
unzip NewDataSet.zip
unzip OldDataSet.zip
```

### get codellama embedding result

Using [Ollama](http://ollama.ai) to deploy `codellama:7b` to get `codellama` embedding results.

After deploy `codellama:7b` locally, run the command below:

```bash
cd codellama
sh embed.sh
```

### get RetriGen dataset

using the following command (taking NewDataSet as an example, OldDataSet needs to replace the dataset path in the script):

```bash
# using IR to get token-based retrieval result
cd IR
mkdir -p result/{NewDataSet,OldDataSet}/codebase_train_query_{train,test,val}
sh codebase_train_query_train.sh
sh codebase_train_query_test.sh
sh codebase_train_query_val.sh

# using codellama to get embedding-based retrieval result
cd codellama
mkdir -p result/{NewDataSet,OldDataSet}/codebase_train_query_{train,test,val}
sh codebase_train_query_train.sh
sh codebase_train_query_test.sh
sh codebase_train_query_val.sh

# using hybrid to get RetriGen dataset
cd hybrid
mkdir -p result/{NewDataSet,OldDataSet}/{train,test,val}
sh test_alpha_0.5.sh
sh train_alpha_0.5.sh
sh val_alpha_0.5.sh
```

## Train and test CodeT5 model to get RetriGen result

run the following command:

```bash
mkdir -p result/{NewDataSet,OldDataSet}
sh run-RetriGen.sh
```
# RQ2

## get RetriGen-token and RetriGen-embed dataset

run the following command to get `RetiGen-token` and `RetriGen-embed` training dataset

```bash
sh get-RetriGen-token-dataset.sh
sh get-RetriGen-embed-dataset.sh
```

## reproduce the result

```bash
sh embed/run-RetriGen-embed.sh
sh none/run-RetriGen-none.sh
sh token/run-RetriGen-token.sh
```

after reproduce the result, you can use RQ1 code to calc accuracy and CodeBLEU
import json
import os

import pandas as pd

for idx in range(4, 7):
    data_dir = f"data/evosuite_buggy_tests/{idx}"

    df = pd.read_csv(os.path.join(data_dir, "assert_model_inputs.csv"))

    with open(os.path.join(data_dir, "retrieval_sources.jsonl"), "r", encoding="utf-8") as f:
        new_sources = [json.loads(line)["source"] for line in f.readlines()]

    df["source"] = new_sources

    df.to_csv(os.path.join(data_dir, "retrieval_assert_model_inputs.csv"), index=False)

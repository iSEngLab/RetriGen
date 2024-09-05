from dataclasses import dataclass, field
from typing import List, Dict
import json
import pandas as pd


@dataclass
class RetrievalArguments:
    codebase_data_path: str = field(default=None, metadata={"help": "codebase data file path"})
    query_data_path: str = field(default=None, metadata={"help": "query data file path"})
    output_dir: str = field(default=None, metadata={"help": "output dir"})
    output_filename: str = field(default=None, metadata={"help": "output filename"})
    search_self: bool = field(default=False, metadata={"help": "search self"})
    batch_size: int = field(default=100, metadata={"help": "batch size"})
    cpu_count: int = field(default=5, metadata={"help": "number of CPU cores to use"})


def read_examples(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        examples = [json.loads(line.strip()) for line in file.readlines()]

    return examples


def read_csv_examples(file_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(file_path)
    examples = []

    for index, row in df.iterrows():
        examples.append({
            "source": row["source"],
            "target": row["target"],
        })

    return examples

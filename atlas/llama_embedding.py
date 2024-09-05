import json
import os
import logging
import transformers
from dataclasses import dataclass, field
from tqdm import tqdm
from configs import read_csv_examples
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingArguments:
    file_path: str = field()
    batch_size: int = field()
    output_dir: str = field()
    output_filename: str = field()


class BatchParam:

    def __init__(self, query_list: List[Dict[str, str]], batch_id: int):
        self.query_list = query_list
        self.batch_id = batch_id


class Example:

    def __init__(self, source: str, target: str, embedding: List[float]):
        self.source = source
        self.target = target
        self.embedding = embedding


def embed(param: BatchParam, embeddings: OllamaEmbeddings) -> List[Example]:
    query_list = param.query_list
    sources = [query["source"] for query in query_list]

    embedding_results = embeddings.embed_documents(sources)
    assert len(sources) == len(embedding_results)
    return [Example(query["source"], query["target"], embedding) for query, embedding in
            zip(query_list, embedding_results)]


def get_param_generator(examples: List[Dict[str, str]], batch_size: int):
    return [BatchParam(examples[i:i + batch_size], i // batch_size) for i in
            range(0, len(examples), batch_size)]


def main():
    parser = transformers.HfArgumentParser(EmbeddingArguments)
    embedding_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if not os.path.exists(embedding_args.output_dir):
        os.makedirs(embedding_args.output_dir)

    examples = read_csv_examples(embedding_args.file_path)
    batch_size = embedding_args.batch_size

    params = get_param_generator(examples, batch_size)
    embeddings = OllamaEmbeddings(
        model="codellama:7b-code-fp16"
    )

    logger.info("****** Start Embedding ******")
    with open(os.path.join(embedding_args.output_dir, f"{embedding_args.output_filename}.jsonl"), "w",
              encoding="utf-8") as f:
        for param in tqdm(params, total=len(params), desc="Embedding..."):
            results = embed(param, embeddings)
            for result in results:
                tmp_dict = {
                    "source": result.source,
                    "target": result.target,
                    "embedding": result.embedding
                }
                f.write(json.dumps(tmp_dict) + "\n")


if __name__ == "__main__":
    main()

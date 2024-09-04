import logging
import time
import json
import math
import os
import transformers
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from typing import List
from configs import RetrievalArguments, read_examples

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchParam:
    def __init__(self,
                 query_sources: List[str],
                 query_embeddings: List[List[float]],
                 batch_id: int):
        self.query_sources = query_sources
        self.query_embeddings = query_embeddings
        self.batch_id = batch_id


def calc_similarities_batch(param: BatchParam,
                            tokenized_corpus: List[List[str]],
                            normalized_codebase_embeddings: np.ndarray,
                            retrieval_args: RetrievalArguments) -> List[List[int]]:
    """ Return top-100 indexes
    :param param: BatchParam
    :param tokenized_corpus: List of tokenized corpus
    :param normalized_codebase_embeddings: Normalized codebase embeddings
    :param retrieval_args: RetrievalArguments

    """
    # calc IR scores
    ir_results = []
    for idx, query in tqdm(enumerate(param.query_sources),
                           desc=f"Calculating {param.batch_id} IR similarities...",
                           leave=False, total=len(param.query_sources), position=1):
        actual_id = param.batch_id * retrieval_args.batch_size + idx
        ir_results.append(calc_ir_scores(tokenized_corpus, query, actual_id, retrieval_args))

    # calc embeddings scores
    query_embeddings = normalize(np.array(param.query_embeddings), norm="l2", axis=1)
    embedding_results = calc_embedding_scores(normalized_codebase_embeddings, query_embeddings, param.batch_id,
                                              retrieval_args)

    assert len(embedding_results) == len(ir_results)

    top_100_results = []
    for embedding_result, ir_result in tqdm(zip(embedding_results, ir_results),
                                            desc=f"Merging {param.batch_id} results...",
                                            leave=False, total=len(ir_results), position=1):
        # Need to normalize cos_sim to [0, 1]
        merged_result = [(1 + cos_sim) / 2 + ir_sim for cos_sim, ir_sim in zip(embedding_result, ir_result)]
        sorted_merged_result = sorted(enumerate(merged_result), key=lambda x: x[1], reverse=True)[:100]
        top_100_results.append([idx for idx, _ in sorted_merged_result])

    return top_100_results


def get_jaccard(list1, list2) -> float:
    intersection = len(set(list1) & set(list2))
    union = len(set(list1) | set(list2))
    return intersection / union


def calc_ir_scores(tokenized_corpus: List[List[str]],
                   query: str,
                   actual_id: int,
                   retrieval_args: RetrievalArguments) -> List[float]:
    tokenized_query = tokenize(query)
    similarities = []
    for j in range(0, len(tokenized_corpus)):
        jaccard: float = get_jaccard(tokenized_query, tokenized_corpus[j])
        similarities.append(jaccard)
    # 如果是 search_self 模式，则将自己的相似度设置为 0
    if retrieval_args.search_self:
        similarities[actual_id] = 0.0
    max_idx, _ = max(enumerate(similarities), key=lambda x: x[1])
    if retrieval_args.search_self:
        assert max_idx != actual_id
    return similarities


def calc_embedding_scores(codebase_embeddings: np.ndarray,
                          query_embeddings: np.ndarray,
                          batch_id: int,
                          retrieval_args: RetrievalArguments) -> List[List[float]]:
    total_similarities = cosine_similarity(np.array(query_embeddings), codebase_embeddings).tolist()
    total_similarities = [[round(float(similarity), 8) for similarity in similarities] for similarities in
                          total_similarities]
    results = []
    for idx, similarities in enumerate(total_similarities):
        actual_id = batch_id * retrieval_args.batch_size + idx
        if retrieval_args.search_self:
            # 将 similarities 中对应的自身 idx 设置为 0
            similarities[actual_id] = 0
        max_idx, _ = max(enumerate(similarities), key=lambda x: x[1])
        if retrieval_args.search_self:
            assert max_idx != actual_id
        results.append(similarities)

    return results


def get_param_generator(query_sources, query_embedding, batch_size):
    return [BatchParam(query_sources[i:i + batch_size], query_embedding[i:i + batch_size], i // batch_size) for i in
            range(0, len(query_embedding), batch_size)]


def tokenize(code: str) -> List[str]:
    return code.split(" ")


def main():
    parser = transformers.HfArgumentParser(RetrievalArguments)
    retrieval_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    retrieval_args.search_self = retrieval_args.codebase_data_path == retrieval_args.query_data_path

    logger.info(retrieval_args)

    if not os.path.exists(retrieval_args.output_dir):
        os.makedirs(retrieval_args.output_dir)

    start_time = time.time()
    codebase_examples = read_examples(retrieval_args.codebase_data_path)
    query_examples = read_examples(retrieval_args.query_data_path)

    # Need to tokenize codebase in advance
    codebase_sources = [example["source"] for example in codebase_examples]
    tokenized_corpus = [tokenize(doc) for doc in
                        tqdm(codebase_sources, desc="split codebase source", total=len(codebase_sources))]
    codebase_embeddings = np.array([example["embedding"] for example in codebase_examples])
    # normalize embeddings
    normalized_codebase_embeddings = normalize(codebase_embeddings, norm="l2", axis=1)
    query_sources = [example["source"] for example in query_examples]
    query_embeddings = [example["embedding"] for example in query_examples]

    logger.info("****** Init tokenized corpus in search-self mode ******"
                if retrieval_args.search_self else "****** Init tokenized corpus in none-search-self mode ******")

    batch_size = retrieval_args.batch_size
    correct = 0
    tasks = []
    params = get_param_generator(query_sources, query_embeddings, batch_size)
    for param in params:
        tasks.append(
            delayed(calc_similarities_batch)(param, tokenized_corpus, normalized_codebase_embeddings, retrieval_args))

    results = Parallel(n_jobs=retrieval_args.cpu_count if retrieval_args.cpu_count else -1, prefer="processes")(
        tqdm(tasks, total=len(tasks), desc="Processing batches")
    )
    assert math.ceil(len(query_examples) / batch_size) == len(results)
    with open(os.path.join(retrieval_args.output_dir, f"{retrieval_args.output_filename}"), "w",
              encoding="utf-8") as output_file:
        for batch_idx, batch_results in tqdm(enumerate(results), total=len(results), desc="Handling results..."):
            batch_examples = query_examples[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            for example, top_100_result in zip(batch_examples, batch_results):
                searched_targets = [codebase_examples[idx]["target"] for idx in top_100_result]

                correct += int(example["target"] == searched_targets[0])
                tmp_dict = {
                    "source": example["source"],
                    "target": example["target"],
                    "searched_targets": searched_targets
                }
                output_file.write(json.dumps(tmp_dict) + "\n")

    logger.info(f"finish search, accuracy: {correct * 100 / len(query_examples)}")
    end_time = time.time()
    logger.info(f"spend time(seconds): {end_time - start_time}")


if __name__ == '__main__':
    main()

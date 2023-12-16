import argparse
import json
import logging
import math
import os
import time
from collections import namedtuple
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

Similarity = namedtuple('Similarity', ['query_idx', 'similarity'])
Result = namedtuple('Result',
                    ['query_source', 'query_target', 'codebase_source', 'codebase_target', 'similarity', 'is_same'])
logger = logging.getLogger(__name__)

# global codebase embeddings
codebase_embeddings = None


class BatchParam:
    def __init__(self, query_embeddings: List[List[float]], batch_id: int):
        self.query_embeddings = query_embeddings
        self.batch_id = batch_id


def calc_cos_sim_multi(param: BatchParam, args):
    query_embeddings = param.query_embeddings
    batch_id = param.batch_id

    total_similarities = cosine_similarity(np.array(query_embeddings), codebase_embeddings).tolist()
    total_similarities = [[round(float(similarity), 8) for similarity in similarities] for similarities in total_similarities]
    results = []
    for idx, similarities in enumerate(total_similarities):
        actual_id = batch_id * args.batch_size + idx
        if args.search_self:
            # 将 similarities 中对应的自身 idx 设置为 0
            similarities[actual_id] = 0
        max_idx, max_value = max(enumerate(similarities), key=lambda x: x[1])
        if args.search_self:
            assert max_idx != actual_id
        results.append([similarities, max_idx, max_value])

    return results


def init_result_tuple():
    total_sim_list = []
    total_result_list = []
    return total_sim_list, total_result_list


def evaluate(args):
    results = pd.read_csv(args.output_result_path)
    total = len(results)
    correct = results["is_same"].sum()

    logger.info(f"accuracy {correct * 1.0 / total}")


def get_param_generator(query_embedding, batch_size):
    for i in range(0, len(query_embedding), batch_size):
        param = BatchParam(query_embedding[i:i + batch_size], i // batch_size)
        yield param


def main(args):
    start_time = time.time()
    codebase_df = pd.read_csv(args.codebase_path)
    query_df = pd.read_csv(args.query_data_path)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    global codebase_embeddings
    codebase_embeddings = np.array([json.loads(embedding) for embedding in codebase_df["embedding"]])
    logger.info("*** Init tokenized corpus in search-self mode *** "
                if args.search_self else "*** Init tokenized corpus in none-search-self mode *** ")
    query_embeddings = [json.loads(embedding) for embedding in query_df["embedding"]]

    total_sim_list, total_result_list = init_result_tuple()
    batch_size = args.batch_size
    storage_size = args.storage_size

    param_generator = get_param_generator(query_embeddings, batch_size)
    partial_function = partial(calc_cos_sim_multi, args=args)
    with Pool(processes=args.cpu_count if args.cpu_count else cpu_count()) as pool:
        results = pool.imap(partial_function, param_generator)

        for idx, result in tqdm(enumerate(results), desc="handle batch",
                               total=math.ceil(len(query_embeddings) / batch_size)):
            for i, [similarities, max_idx, max_value] in enumerate(result):
                actual_idx = idx * batch_size + i
                total_sim_list.append(Similarity(query_idx=actual_idx,
                                                 similarity=similarities))

                query_target = query_df["target"][actual_idx]
                codebase_target = codebase_df["target"][max_idx]
                total_result_list.append(Result(query_source=query_df["source"][actual_idx],
                                                query_target=query_target,
                                                codebase_source=codebase_df["source"][max_idx],
                                                codebase_target=codebase_target,
                                                similarity=max_value,
                                                is_same=1 if query_target == codebase_target else 0))

            if ((idx + 1) * batch_size) % storage_size == 0:
                logger.info("*** Write partial result to file *** ")
                append_header = False
                if not os.path.exists(args.output_result_path):
                    append_header = True
                pd.DataFrame(total_sim_list, columns=Similarity._fields).to_parquet(
                    args.output_sim_path.format(((idx + 1) * batch_size) // storage_size), index=False)
                pd.DataFrame(total_result_list, columns=Result._fields).to_csv(args.output_result_path, mode="a",
                                                                               header=append_header,
                                                                               index=False)
                del total_sim_list
                del total_result_list
                total_sim_list, total_result_list = init_result_tuple()

    # 当最后一个 batch 不足 storage_size 时，需要额外的存储
    if ((idx + 1) * batch_size) % storage_size != 0:
        logger.info("*** Write partial result to file *** ")
        append_header = False
        if not os.path.exists(args.output_result_path):
            append_header = True
        pd.DataFrame(total_sim_list, columns=Similarity._fields).to_parquet(
            args.output_sim_path.format(((idx + 1) * batch_size) // storage_size + 1),
            index=False)
        pd.DataFrame(total_result_list, columns=Result._fields).to_csv(args.output_result_path, mode="a", header=append_header,
                                                                       index=False)

    logger.info("finish search")
    end_time = time.time()
    logger.info(f"spend time(seconds): {end_time - start_time}")
    evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--codebase_path", default="./data/assert_train_new.csv", type=str, required=False,
                        help="codebase file path.")
    parser.add_argument("--query_data_path", default="./data/assert_test_new.csv", type=str, required=False,
                        help="query data file path.")
    parser.add_argument("--output_sim_path", default="./result/total_sim_{}.parquet", type=str, required=False,
                        help="output sim path.")
    parser.add_argument("--output_result_path", default="./result/total_result.csv", type=str, required=False,
                        help="output result path.")
    parser.add_argument("--search_self", default=False, type=bool, required=False,
                        help="whether codebase is same as query data or not")
    parser.add_argument("--batch_size", default=100, type=int, required=False,
                        help="batch size to search codebase in a child process")
    parser.add_argument("--storage_size", default=1000, type=int, required=False,
                        help="storage size to store partial result")
    parser.add_argument("--cpu_count", type=int, required=False,
                        help="cpu count to start child progresses")

    args = parser.parse_args()
    args.search_self = args.codebase_path == args.query_data_path
    if args.search_self:
        logger.info("search codebase self")
    main(args)

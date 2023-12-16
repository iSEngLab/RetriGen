import argparse
import logging
from typing import List, Union
from tqdm import tqdm
import os
import pandas as pd
import json
from collections import namedtuple
import re
from multiprocessing import Pool, cpu_count

Result = namedtuple('Result',
                    ['query_source', 'query_target', 'codebase_source', 'codebase_target', 'similarity', 'is_same'])
Dataset = namedtuple('Dataset', ['source', 'target'])
logger = logging.getLogger(__name__)


class FilenameParam:

    def __init__(self, codellama_filename, IR_filename):
        self.codellama_filename = codellama_filename
        self.IR_filename = IR_filename


def merge_similarities(param: FilenameParam):
    codellama_filename = param.codellama_filename
    IR_filename = param.IR_filename
    codellama_similarities = read_similarities_from_file(codellama_filename)
    IR_similarities = read_similarities_from_file(IR_filename)
    results = []

    for codellama_similarity_tuple, IR_similarity_tuple in tqdm(zip(codellama_similarities, IR_similarities),
                                                                desc="merge similarities",
                                                                total=len(codellama_similarities),
                                                                leave=False,
                                                                position=1):
        result = merge_similarity(codellama_similarity_tuple, IR_similarity_tuple)
        results.append(result)

    return results


def merge_similarity(codellama_similarity_tuple, IR_similarity_tuple):
    codellama_query_idx = codellama_similarity_tuple[0]
    codellama_similarities = codellama_similarity_tuple[1].tolist()
    IR_query_idx = IR_similarity_tuple[0]
    IR_similarities = IR_similarity_tuple[1]

    # for validate
    if codellama_query_idx != IR_query_idx:
        err(f"codellama query idx should equal to IR query idx, "
            f"codellama query idx: {codellama_query_idx}, IR query idx: {IR_query_idx}")
    if len(codellama_similarities) != len(IR_similarities):
        err(f"codellama similarities length should equal to IR similarities length, "
            f"codellama query idx: {codellama_query_idx}, IR query idx: {IR_query_idx}")

    # merge similarities
    hybrid_similarities = [args.alpha * float(IR_similarity) + (1 - args.alpha) * float(codellama_similarity) for
                           IR_similarity, codellama_similarity in
                           zip(IR_similarities, codellama_similarities)]
    max_idx, max_value = max(enumerate(hybrid_similarities), key=lambda x: x[1])
    return [hybrid_similarities, max_idx, max_value, codellama_query_idx, IR_query_idx]


def err(msg: str):
    logger.error(msg)
    raise ValueError(msg)


def handle_new_dataset(query_source: str,
                       searched_target: str):
    substring = '"<AssertPlaceHolder>"'
    assert_placeholder_idx = query_source.find(substring)
    if assert_placeholder_idx != -1:
        split_idx = assert_placeholder_idx + len(substring)
        new_source = query_source[:split_idx] + f" /* {searched_target} */" + query_source[split_idx:]
        return new_source
    else:
        logger.info("do not found <AssertPlaceHolder>")

    return None


def evaluate(total: int, correct: int):
    logger.info(f"accuracy {correct * 1.0 / total}")


def read_similarities_from_file(filename: str) -> List[List[Union[int, List[float]]]]:
    # 读取 parquet 文件并存储到 DataFrame 中
    sim_df = pd.read_parquet(os.path.join(filename))
    try:
        similarities = [[data["query_idx"], json.loads(data["similarity"])]
                        for _, data in sim_df.iterrows()]
    except Exception:
        similarities = [[data["query_idx"], data["similarity"]]
                        for _, data in sim_df.iterrows()]

    return similarities


def find_number_in_str(content: str):
    result = re.findall(r'\d+', content)
    if result:
        return int(result[-1])

    logger.info(f"can't find number in filename, content={content}")


def read_file_names_from_dir(folders: List[str], args) -> List[str]:
    filenames = []
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".parquet"):
                filenames.append(os.path.join(folder, file))

    filenames = sorted(filenames, key=find_number_in_str)
    if args.skip_file:
        logger.info(f"*** Skip file count {args.skip_file} ***")
        filenames = filenames[args.skip_file:]
    if args.file_length:
        logger.info(f"*** Total file length {args.file_length} ***")
        filenames = filenames[:args.file_length]
    return filenames


def get_param_generator(codellama_filenames, IR_filenames):
    for codellama_filename, IR_filename in zip(codellama_filenames, IR_filenames):
        yield FilenameParam(codellama_filename, IR_filename)


def main(args):
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    codellama_sim_path = args.codellama_sim_path
    IR_sim_path = args.IR_sim_path
    # 文件名列表
    codellama_filenames = read_file_names_from_dir(codellama_sim_path, args)
    IR_filenames = read_file_names_from_dir(IR_sim_path, args)
    logger.info(f"get filenames, codellama={codellama_filenames}, IR_filenames={IR_filenames}")
    # 读取 codebase
    codebase_df = pd.read_csv(args.codebase_dataset)
    query_df = pd.read_csv(args.query_dataset)
    codebase_source = codebase_df["source"]
    query_source = query_df["source"]

    if len(codellama_filenames) != len(IR_filenames):
        err("codellama_sim_path and IR_sim_path should have the same amount of files and same rows of a file")

    # 合并相似度
    param_generator = get_param_generator(codellama_filenames, IR_filenames)
    total_result_list = []
    new_dataset_list = []
    correct = 0
    total = 0
    with Pool(processes=args.cpu_count if args.cpu_count else cpu_count()) as pool:
        results = pool.imap(merge_similarities, param_generator)
        for idx, result in tqdm(enumerate(results), desc="handle result", total=len(codellama_filenames)):
            for i, [_, max_idx, max_value, codellama_query_idx, _] in enumerate(result):
                query_target = query_df["target"][codellama_query_idx]
                codebase_target = codebase_df["target"][max_idx]
                total += 1
                if query_target == codebase_target:
                    correct += 1

                # save result and dataset
                if args.save_result:
                    total_result_list.append(Result(query_source=query_source[codellama_query_idx],
                                                    query_target=query_target,
                                                    codebase_source=codebase_source[max_idx],
                                                    codebase_target=codebase_target,
                                                    similarity=max_value,
                                                    is_same=1 if query_target == codebase_target else 0))
                if args.save_dataset:
                    # 保存处理后的 dataset
                    new_source = handle_new_dataset(query_source[codellama_query_idx], codebase_target)
                    if new_source is not None:
                        new_dataset = Dataset(source=new_source, target=query_target)
                        new_dataset_list.append(new_dataset)
            if args.save_result and ((idx + 1) * args.batch_size) % args.storage_size == 0:
                logger.info("*** Write partial result to file *** ")
                append_header = False
                if not os.path.exists(args.output_result_path):
                    append_header = True
                pd.DataFrame(total_result_list, columns=Result._fields).to_csv(args.output_result_path,
                                                                               mode="a",
                                                                               header=append_header,
                                                                               index=False)
                del total_result_list
                total_result_list = []

            if args.save_dataset and ((idx + 1) * args.batch_size) % args.storage_size == 0:
                logger.info("*** Write partial Dataset to file ***")
                append_header = False
                if not os.path.exists(args.output_dataset_path):
                    append_header = True
                pd.DataFrame(new_dataset_list, columns=Dataset._fields).to_csv(args.output_dataset_path,
                                                                               mode="a",
                                                                               header=append_header,
                                                                               index=False)
                del new_dataset_list
                new_dataset_list = []

    if args.save_result and ((idx + 1) * args.batch_size) % args.storage_size != 0:
        logger.info("*** Write partial result to file *** ")
        append_header = False
        if not os.path.exists(args.output_result_path):
            append_header = True
        pd.DataFrame(total_result_list, columns=Result._fields).to_csv(args.output_result_path, mode="a",
                                                                       header=append_header,
                                                                       index=False)
    if args.save_dataset and ((idx + 1) * args.batch_size) % args.storage_size != 0:
        logger.info("*** Write partial Dataset to file ***")
        append_header = False
        if not os.path.exists(args.output_dataset_path):
            append_header = True
        pd.DataFrame(new_dataset_list, columns=Dataset._fields).to_csv(args.output_dataset_path, mode="a",
                                                                       header=append_header,
                                                                       index=False)

    evaluate(total, correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--codellama_sim_path", nargs="+", type=str, required=True,
                        help="codellama similarity file path")
    parser.add_argument("--IR_sim_path", nargs="+", type=str, required=True,
                        help="IR similarity file path")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="alpha to merge similarity, "
                             "hybrid_similarity = ${alpha}IR_similarity + ${1 - alpha}codellama_similarity")

    parser.add_argument("--save_dataset", action='store_true', default=False,
                        help="whether to save dataset")
    parser.add_argument("--save_result", action="store_true", default=False,
                        help="whether to save result")
    parser.add_argument("--query_dataset", type=str, help="raw train dataset to add hybrid target")
    parser.add_argument("--codebase_dataset", type=str, required=True,
                        help="codebase dataset to get target assertion")
    parser.add_argument("--output_dataset_path", type=str, default="./assert_train_new_hybrid.csv",
                        help="save dataset output path, need --save_dataset is True")
    parser.add_argument("--output_result_path", type=str, default="./total_result.csv",
                        help="output result file path")
    parser.add_argument("--storage_size", default=1000, type=int, required=False,
                        help="storage size to store partial result")
    parser.add_argument("--cpu_count", type=int, required=False,
                        help="cpu count to start child progresses")
    parser.add_argument("--batch_size", type=int, default=1000, help="row amount in a file")
    parser.add_argument("--skip_file", type=int, help="skip file count")
    parser.add_argument("--file_length", type=int, help="file length")

    args = parser.parse_args()
    main(args)

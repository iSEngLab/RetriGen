from __future__ import absolute_import

from argparse import Namespace
import os
import torch
import random
import logging
import numpy as np
import pandas as pd
import re
from model import DataBase, build_model
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset
from typing import List
from args_config import add_args

from transformers import (PreTrainedTokenizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compare_strings_ignore_whitespace(str1, str2):
    # 移除字符串中的空格
    str1_without_space = re.sub(r'\s', '', str1)
    str2_without_space = re.sub(r'\s', '', str2)

    # 使用re.match()函数进行匹配比较
    if re.match(f'^{re.escape(str1_without_space)}$', str2_without_space):
        return True
    else:
        return False


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename: str) -> List[Example]:
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="read examples"):
        source = row["source"]
        target = row["target"]
        examples.append(
            Example(
                idx=idx,
                source=source,
                target=target
            )
        )

    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 query_ids
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.query_ids = query_ids


def convert_examples_to_features(examples: List[Example], tokenizer: PreTrainedTokenizer, args: Namespace,
                                 stage: str = None):
    """convert examples to token ids"""
    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples), desc="convert examples to features"):
        # source
        source_str = example.source
        source_tokens = tokenizer.tokenize(source_str)
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens[:args.code_length])
        query_tokens = [tokenizer.cls_token] + source_tokens[:args.code_length - 2] + [tokenizer.eos_token]
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        padding_length = args.code_length - len(query_ids)
        query_ids += [tokenizer.pad_token_id] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_str = example.target
            target_tokens = tokenizer.tokenize(target_str)[:args.nl_length]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                query_ids
            )
        )
    return features


class MyDataset(Dataset):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def build_index(self, retriever, args):
        with torch.no_grad():
            inputs = [feature.query_ids for feature in self.features]
            inputs = torch.tensor(inputs, dtype=torch.long)
            dataset = TensorDataset(inputs)
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
            query_vecs = []
            retriever.eval()
            for batch in tqdm(dataloader, desc="build index", total=len(dataloader)):
                code_inputs = torch.tensor(batch[0]).to(args.device)
                code_vec = retriever(code_inputs)
                query_vecs.append(code_vec.cpu().numpy())
            query_vecs = np.concatenate(query_vecs, 0)
            index = DataBase(query_vecs)
        return index


def do_nothing_collator(batch):
    return batch


def main():
    args = add_args()

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    logger_path = os.path.join(args.output_dir, 'train.log') if args.do_train else os.path.join(args.output_dir,
                                                                                                'test.log')
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    config, generator, retriever, tokenizer = build_model(args)

    logger.info("Training/evaluation parameters %s", args)
    generator.to(args.device)
    retriever.to(args.device)
    if args.n_gpu > 1:
        generator = torch.nn.DataParallel(generator)
        retriever = torch.nn.DataParallel(retriever)

    prefix = [tokenizer.cls_token_id]
    postfix = [tokenizer.sep_token_id]
    # Use // to comment Java code
    sep = tokenizer.convert_tokens_to_ids(["\n", "//"])
    sep_ = tokenizer.convert_tokens_to_ids(["\n"])

    def cat_to_input(code, similar_assertion, similar_code):
        new_input = code + sep + similar_assertion + sep_ + similar_code
        new_input = prefix + new_input[:args.max_source_length - 2] + postfix
        padding_length = args.max_source_length - len(new_input)
        new_input += padding_length * [tokenizer.pad_token_id]
        return new_input

    def cat_to_output(assertion):
        output = prefix + assertion[:args.max_target_length - 2] + postfix
        padding_length = args.max_target_length - len(output)
        output += padding_length * [tokenizer.pad_token_id]
        return output

    if args.do_test:
        # load retriever
        checkpoint_prefix = 'checkpoint-best-bleu/retriever_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = retriever.module if hasattr(retriever, 'module') else retriever
        model_to_load.load_state_dict(torch.load(output_dir))

        # load generator
        checkpoint_prefix = 'checkpoint-best-bleu/generator_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = generator.module if hasattr(generator, 'module') else generator
        model_to_load.load_state_dict(torch.load(output_dir))

        # Build retrival base
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args)
        train_dataset = MyDataset(train_features)
        index = train_dataset.build_index(retriever, args)

        test_examples = read_examples(args.test_filename)
        test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
        test_data = MyDataset(test_features)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size,
                                     collate_fn=do_nothing_collator)

        logger.info("***** Running Testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)

        retriever.eval()
        generator.eval()
        p = []
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Running Testing"):
            query = [feature.query_ids for feature in batch]
            query = torch.tensor(query, dtype=torch.long).to(device)
            query_vec = retriever(query)
            query_vec_cpu = query_vec.detach().cpu().numpy()
            i = index.search(query_vec_cpu, 1)
            inputs = []
            for no, feature in enumerate(batch):
                relevant = train_dataset.features[i[no][0]]
                inputs.append(cat_to_input(feature.source_ids, relevant.target_ids, relevant.source_ids))
            with torch.no_grad():
                inputs = torch.tensor(inputs, dtype=torch.long).to(device)
                source_mask = inputs.ne(tokenizer.pad_token_id)
                preds = generator(inputs,
                                  attention_mask=source_mask,
                                  is_generate=True)
                top_preds = list(preds.cpu().numpy())
                p.extend(top_preds)
        p = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in p]
        generator.train()
        retriever.train()

        predictions = []
        for pred, gold in tqdm(zip(p, test_examples)):
            predictions.append(pred)

        df = pd.read_csv(args.test_filename)
        df['assert_pred'] = predictions
        targets = df['target']
        match = []
        for p, t in zip(predictions, targets):
            try:
                if compare_strings_ignore_whitespace(p, t):
                    match.append(1)
                else:
                    match.append(0)
            except Exception as e:
                logger.error(f"encounter error {e} with {p} and {t}")
                match.append(0)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        df['match'] = match
        df.drop('source', axis=1).to_csv(os.path.join(args.output_dir, 'assertion_preds.csv'), index=False)


if __name__ == '__main__':
    main()

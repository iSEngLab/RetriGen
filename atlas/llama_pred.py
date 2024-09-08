import json
import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from tqdm import tqdm

import torch
import transformers
from datasets import load_dataset

logger = logging.getLogger(__name__)

PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Implement an assertion statement to replace "<AssertPlaceHolder>". The assertion needs to test the focal method for correctness. The test case and focal method is:

{instruction}

@@ Response
"""


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the data directory"}
    )
    test_filename: str = field(
        default=None,
        metadata={"help": "Test data filename."}
    )
    cache_dir: str = field(
        default="./cache",
        metadata={"help": "Directory to store cached data."}
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes to use for data preprocessing."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: torch.dtype = field(default=torch.bfloat16)
    device_map: str = field(default="auto")


@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=256)
    max_length: int = field(default=1024)
    num_beams: int = field(default=10)
    num_return_sequences: int = field(default=1)
    device: str = field(default="cuda")
    n_gpu: int = field(default=1)
    output_dir: str = field(default="./saved_models")
    use_instruct: bool = field(default=False)


def get_test_data(data_args) -> Dict:
    data_files = {}
    if data_args.test_filename is not None:
        data_files["test"] = data_args.test_filename

    return load_dataset("json", data_dir=data_args.data_path,
                        data_files=data_files, cache_dir=data_args.cache_dir)["test"]


def build_model(model_args: "ModelArguments") -> transformers.PreTrainedModel:
    return transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        device_map=model_args.device_map
    )


def build_tokenizer(model_args: "ModelArguments") -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=True,
    )
    logger.info(f"tokenizer pad_token {tokenizer.pad_token}, tokenizer pad_token_id: {tokenizer.pad_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    return tokenizer


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, generation_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    logger.warning(
        f"device: {generation_args.device}, n_gpu: {generation_args.n_gpu}"
    )
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Generation parameters {generation_args}")

    model = build_model(model_args)
    tokenizer = build_tokenizer(model_args)
    dataset = get_test_data(data_args=data_args)
    model.to(generation_args.device)
    predict_results = []
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating..."):
        tmp_dict = {}
        source = sample["source"]
        if generation_args.use_instruct:
            source = PROMPT.format(instruction=source)
        inputs = tokenizer(source, return_tensors='pt', truncation=True, max_length=generation_args.max_length)
        inputs_len = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids.to(generation_args.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_args.max_new_tokens,
                num_return_sequences=generation_args.num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=generation_args.num_beams
            )
        output_ids = outputs[:, inputs_len:]
        output_diff = tokenizer.batch_decode(output_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
        tmp_dict['predict'] = output_diff[0] if len(output_diff) == 1 else output_diff
        tmp_dict['label'] = tokenizer.decode(tokenizer.encode(sample['target']), skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)

        predict_results.append(tmp_dict)

    output_path = os.path.join(generation_args.output_dir, "generated_predictions.jsonl")
    with open(output_path, "w", encoding="utf-8") as output_file:
        for idx, predict_result in enumerate(predict_results):
            tmp_dict = {
                "predict": predict_result["predict"],
                "label": predict_result["label"]
            }
            output_file.write(json.dumps(tmp_dict) + "\n")


if __name__ == "__main__":
    main()

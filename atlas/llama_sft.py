import logging
import sys
import os
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import datasets
import transformers
import matplotlib.pyplot as plt
from transformers import (
    PreTrainedTokenizer,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_dataset
from functools import partial

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"

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
    train_filename: str = field(
        default=None,
        metadata={"help": "Training data filename."}
    )
    eval_filename: str = field(
        default=None,
        metadata={"help": "Evaluation data filename."}
    )
    cache_dir: str = field(
        default="./cache",
        metadata={"help": "Directory to store cached data."}
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes to use for data preprocessing."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: torch.dtype = field(default=torch.bfloat16)
    device_map: str = field(default="auto")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./saved_models")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "The maximum input sequence length after tokenization."}
    )
    num_train_epochs: float = field(default=75)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    logging_steps: int = field(default=1)
    eval_strategy: str = field(default="epoch")
    eval_steps: int = field(default=1000)
    save_strategy: str = field(default="epoch")
    save_steps: int = field(default=1000)
    report_to: str = field(default="none")
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)
    save_total_limit: int = field(default=10)
    seed: int = field(default=42)
    use_instruct: bool = field(default=False)
    early_stop_patience: int = field(
        default=2,
        metadata={"help": "Early stopping patience for early stopping"}
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: List[str] = ["loss"]) -> None:
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


def tokenize(text,
             tokenizer: PreTrainedTokenizer,
             training_args: TrainingArguments,
             add_eos_token: bool = True):
    result = tokenizer(
        text,
        truncation=True,
        max_length=training_args.model_max_length,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < training_args.model_max_length
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= training_args.model_max_length:
        result["input_ids"][training_args.model_max_length - 1] = tokenizer.eos_token_id
        result["attention_mask"][training_args.model_max_length - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def get_prompt_target(sample):
    return sample['source'], sample['target']


def generate_and_tokenize_prompt(sample,
                                 tokenizer: PreTrainedTokenizer,
                                 training_args: TrainingArguments,
                                 stage: str = "train"):
    input_text, target = get_prompt_target(sample)

    # Construct Instruction
    if training_args.use_instruct:
        input_text = PROMPT.format(instruction=input_text)

    full_text = input_text + target
    tokenized_full_text = tokenize(full_text, tokenizer, training_args)
    tokenized_input_text = tokenize(input_text, tokenizer, training_args)
    input_len = len(tokenized_input_text["input_ids"]) - 1
    tokenized_full_text["labels"] = [IGNORE_INDEX] * input_len + tokenized_full_text["labels"][input_len:]
    return tokenized_full_text if stage != "test" else tokenized_input_text


def get_data_module(tokenizer, training_args, data_args) -> Dict:
    data_files = {}
    if data_args.train_filename is not None:
        data_files["train"] = data_args.train_filename
    if data_args.eval_filename is not None:
        data_files["eval"] = data_args.eval_filename

    dataset = load_dataset("json", data_dir=data_args.data_path,
                           data_files=data_files, cache_dir=data_args.cache_dir)

    result = dict()
    if data_args.train_filename is not None:
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args, stage="train"),
            num_proc=data_args.num_proc,
        )
        result["train_dataset"] = train_dataset
    if data_args.eval_filename is not None:
        eval_dataset = dataset["eval"]
        eval_dataset = eval_dataset.map(
            partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args, stage="eval"),
            num_proc=data_args.num_proc,
        )
        result["eval_dataset"] = eval_dataset

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    result["data_collator"] = data_collator
    return result


def build_model(model_args: "ModelArguments") -> transformers.PreTrainedModel:
    return transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        # can't use "auto" in accelerate launch
        device_map=model_args.device_map,
    )


def build_tokenizer(model_args: "ModelArguments",
                    training_args: "TrainingArguments") -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    logger.info(f"tokenizer pad_token {tokenizer.pad_token}, tokenizer pad_token_id: {tokenizer.pad_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    return tokenizer


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    model = build_model(model_args)
    tokenizer = build_tokenizer(model_args, training_args)

    data_module = get_data_module(tokenizer=tokenizer, training_args=training_args, data_args=data_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stop_patience)],
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"]
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(output_dir=training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and training_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


if __name__ == "__main__":
    main()

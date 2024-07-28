

from datasets import DatasetDict, load_dataset, Dataset
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import os
import evaluate
import torch
from tqdm import tqdm

from functools import partial


import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed, pipeline, TextGenerationPipeline
from peft import LoraConfig, TaskType, AutoPeftModelForCausalLM
from trl import SFTTrainer

import sys

import datasets




logger = logging.getLogger()

@dataclass
class ModelArguments:
    model_id: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    subsample: float
    target_modules: List[str]
    max_seq_length: int

def setup_logger(logger):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = "INFO"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_sql_data(random_state: int = 42, subsample: float = None) -> DatasetDict:
    dataset_name = "b-mc2/sql-create-context"
    dataset = load_dataset(dataset_name, split="train")
    print(f"dataset size: {len(dataset)}")
    print(dataset.shuffle()[0])

    if subsample is not None:
        dataset = dataset.shuffle(seed=random_state).select(range(int(len(dataset) * subsample)))
        print(f"dataset new size: {len(dataset)}")
    dataset = dataset.train_test_split(test_size=0.05, seed=random_state)
    return dataset

@torch.no_grad()
def predict(pipe: TextGenerationPipeline, question: str, context: str) -> str:
    messages = [{"content": f"{context}\n Input: {question}", "role": "user"}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180,)
    sql = outputs[0]["generated_text"][len(prompt) :].strip()    
    return sql

def get_pipeline(model_load_path: str) -> TextGenerationPipeline:
    device_map = {"": 0}
    new_model = AutoPeftModelForCausalLM.from_pretrained(
        model_load_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    merged_model = new_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path, trust_remote_code=True
    )
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
    return pipe

def run_evaluate_on_json(json_path: Path, model_load_path: Path):
    df = Dataset.from_json(str(json_path)).to_pandas()
    pipe = get_pipeline(model_load_path=model_load_path)
    rouge = evaluate.load("rouge")

    generated_sql = []
    for idx in tqdm(range(len(df))):
        context = df.iloc[idx]["context"]
        question = df.iloc[idx]["question"]
        sql =  predict(question=question, context=context, pipe=pipe)
        generated_sql.append(sql)

    results = rouge.compute(predictions=generated_sql, references=df["answer"].values)
    print(f"Metrics {results}")



def create_message_column(row: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    messages = []
    user = {"content": f"{row['context']}\n Input: {row['question']}", "role": "user"}
    messages.append(user)
    assistant = {"content": f"{row['answer']}", "role": "assistant"}
    messages.append(assistant)
    return {"messages": messages}


def format_dataset_chatml(row: Dict[str, List[Dict[str, str]]], tokenizer: AutoTokenizer) -> Dict[str, str]:
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


def process_dataset(model_id: str, dataset: DatasetDict) -> DatasetDict:
    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(partial(format_dataset_chatml, tokenizer=tokenizer))
    return dataset_chatml


def get_model(model_id: str, device_map: Dict[str, int]) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
        # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    else:
        compute_dtype = torch.float16
        attn_implementation = "sdpa"
        # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
        print(attn_implementation)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map, attn_implementation=attn_implementation)
    return tokenizer, model


def train(config: Dict[str, Any]):

    setup_logger(logger)

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_dict(config)

    logger.info(f"model_args = {model_args}")
    logger.info(f"training_args = {training_args}")
    set_seed(training_args.seed)

    dataset = get_sql_data(subsample=model_args.subsample)
    dataset_chatml = process_dataset(model_id=model_args.model_id, dataset=dataset)
    logger.info(dataset_chatml["train"][0])

    device_map = {"": 0}
    tokenizer, model = get_model(model_id=model_args.model_id, device_map=device_map)
    peft_config = LoraConfig(r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, task_type=TaskType.CAUSAL_LM, target_modules=model_args.target_modules,)

    trainer = SFTTrainer(model=model, train_dataset=dataset_chatml["train"], eval_dataset=dataset_chatml["test"], peft_config=peft_config, dataset_text_field="text", max_seq_length=model_args.max_seq_length, tokenizer=tokenizer, args=training_args)
    trainer.train()
    trainer.save_model()
    trainer.create_model_card()


def main():
    run_name = os.getenv("RUN_NAME", 'phi-3-mini-lora-text2sql-default')
    config = {
        "num_train_epochs": 1,
        "subsample": None,
        "output_dir": run_name,
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj",],  
        "max_seq_length": 512, 
        "push_to_hub": True,
        "eval_strategy": "steps",
        "eval_strategy": "steps",
        "do_eval": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "optim": "adamw_torch",
        "warmup_ratio": 0.1,
        "logging_first_step": True,
        "logging_steps": 500,
        "save_steps": 500,
        "eval_steps": 500,
        "eval_on_start": True,
        "seed": 42,
        "bf16": True,
        "fp16": False,
        "report_to": ["wandb"],
        "lr_scheduler_type": "linear",
        "log_level" : "debug",
    }
    train(config=config)

if __name__ == '__main__':
    main()
    
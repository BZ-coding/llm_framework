import os
import shutil
import logging

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM

from utils.args import get_train_args, get_lora_args
from utils.data import get_generate_and_tokenize_prompt_fn
from utils.tokenizer import get_tokenizer
from tools.log import get_logger

BASE_MODEL = '/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma'
DATA_PATH = '/mnt/nfs/zsd_server/data/origin/alpaca_data_cleaned_archive.json'
SAVE_PATH = '/mnt/nfs/zsd_server/models/my/llama-7b_save'
project_name = 'clm_with_trainer'

train_args = get_train_args(
    epoch=2.0,  # 0.05 for test
    # save_steps=20,
    # eval_steps=20,
)

lora_args = get_lora_args(
    lora_target_modules=(  # kwargs不能hash list
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
)

if os.path.exists(SAVE_PATH):
    os.system(f"rm -rf {SAVE_PATH}")  # 在nfs上shutil.rmtree会报正忙、非空
os.makedirs(SAVE_PATH, exist_ok=True)

log_level = logging.INFO
logger = get_logger(log_level=log_level, logger_log_level=logging.INFO, log_file=train_args.log_file)

tokenizer = get_tokenizer(tokenizer_path=BASE_MODEL)

data = load_dataset("json", data_files=DATA_PATH)

generate_and_tokenize_prompt = get_generate_and_tokenize_prompt_fn(tokenizer=tokenizer,
                                                                   max_length=train_args.max_length)

data = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)
data["train"] = data["train"].map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names, num_proc=2)
data["test"] = data["test"].map(generate_and_tokenize_prompt, remove_columns=data["test"].column_names, num_proc=2)
logger.info(data)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    return_tensors="pt",
    padding=True,
    pad_to_multiple_of=8,
    # pad_to_multiple_of=ARGS.max_length,  # the max_length arg is unused to padding label
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    # load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",  # "auto"
)

lora_config = LoraConfig(
    r=lora_args.lora_r,
    lora_alpha=lora_args.lora_alpha,
    target_modules=lora_args.lora_target_modules,
    lora_dropout=lora_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.config.use_cache = False

# model = torch.compile(model)
logger.info(model)

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=train_args.micro_batch_size,
    gradient_accumulation_steps=train_args.gradient_accumulation_steps,
    # warmup_steps=100,
    warmup_ratio=0.1,
    num_train_epochs=train_args.epoch,
    learning_rate=train_args.learning_rate,
    # fp16=True,
    bf16=True,
    optim="adamw_torch",
    evaluation_strategy="steps",
    eval_steps=train_args.eval_steps,
    save_strategy="steps",
    save_steps=train_args.save_steps,
    output_dir=SAVE_PATH,
    save_total_limit=3,
    load_best_model_at_end=False,
    report_to=["tensorboard"],
    logging_dir=os.path.join(SAVE_PATH, project_name),
    logging_steps=1,
    auto_find_batch_size=False,
    # torch_compile=True,
    do_train=True,
    overwrite_output_dir=True,
    save_safetensors=True,
)
logger.info(training_arguments)

# batch = data_collator([data["train"][i] for i in range(1, 3)])
# print(batch.keys())


trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=training_arguments,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(SAVE_PATH, safe_serialization=True)
tokenizer.save_pretrained(SAVE_PATH)

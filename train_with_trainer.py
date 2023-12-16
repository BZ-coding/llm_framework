import os
import shutil

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaForCausalLM

from utils.args import get_train_args, get_lora_args
from utils.data import get_generate_and_tokenize_prompt_fn
from utils.tokenizer import get_tokenizer
from tools.log import get_logger

BASE_MODEL = '/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma'
DATA_PATH = '/mnt/nfs/zsd_server/data/origin/alpaca_data_cleaned_archive.json'
SAVE_PATH = '/mnt/nfs/zsd_server/models/my/llama-7b_save'

train_args = get_train_args(
    epoch=2.0  # 0.1 for test
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

logger = get_logger(log_file=train_args.log_file)

tokenizer = get_tokenizer(tokenizer_path=BASE_MODEL)

data = load_dataset("json", data_files=DATA_PATH)

generate_and_tokenize_prompt = get_generate_and_tokenize_prompt_fn(tokenizer=tokenizer,
                                                                   max_length=train_args.max_length)

data = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)
data["train"] = data["train"].map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
data["test"] = data["test"].map(generate_and_tokenize_prompt, remove_columns=data["test"].column_names)
logger.info(data)

model = LlamaForCausalLM.from_pretrained(
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

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

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
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    output_dir=SAVE_PATH,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to=["tensorboard"],
    logging_dir='logs',
    logging_steps=1,
    auto_find_batch_size=False,
    # torch_compile=True,
    do_train=True,
    overwrite_output_dir=True,
    # save_safetensors=True,
)
logger.info(training_arguments)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    return_tensors="pt",
    padding=True,
    pad_to_multiple_of=8,
    # pad_to_multiple_of=ARGS.max_length, 
)

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

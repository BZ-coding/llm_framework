import dataclasses
import os
import shutil
import logging
import math

import torch
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger as accelerate_get_logger
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import get_scheduler, AutoModelForCausalLM
from tqdm.auto import tqdm

from utils.args import get_train_args, get_lora_args
from utils.data import get_generate_and_tokenize_prompt_fn
from utils.tokenizer import get_tokenizer
from tools.log import get_logger

BASE_MODEL = '/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma'
DATA_PATH = '/mnt/nfs/zsd_server/data/origin/alpaca_data_cleaned_archive.json'
SAVE_PATH = '/mnt/nfs/zsd_server/models/my/llama-7b_save'

train_args = get_train_args(
    epoch=2.0  # 0.05 for test
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

accelerator = Accelerator(
    gradient_accumulation_steps=train_args.gradient_accumulation_steps,
    log_with=train_args.report_to,
    project_dir=SAVE_PATH,
)

log_level = logging.WARNING
if accelerator.is_local_main_process:
    log_level = logging.INFO
logger = accelerate_get_logger(__name__)
_ = get_logger(log_level=log_level, logger_log_level=logging.INFO, logger=logger.logger, log_file=train_args.log_file)

logger.info(accelerator.state, main_process_only=False)
accelerator.wait_for_everyone()

tokenizer = get_tokenizer(tokenizer_path=BASE_MODEL)

# todo: add loss mask
with accelerator.main_process_first():
    logger.info("Start handle dataset.", main_process_only=False)
    data = load_dataset("json", data_files=DATA_PATH)  # todo: add use cache

    generate_and_tokenize_prompt = get_generate_and_tokenize_prompt_fn(tokenizer=tokenizer,
                                                                       max_length=train_args.max_length)

    data = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)
    data["train"] = data["train"].map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names,
                                      num_proc=2)
    data["test"] = data["test"].map(generate_and_tokenize_prompt, remove_columns=data["test"].column_names, num_proc=2)
    logger.info(data)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    return_tensors="pt",
    padding=True,
    pad_to_multiple_of=8,
    # pad_to_multiple_of=ARGS.max_length,
)

train_dataloader = DataLoader(data["train"],
                              shuffle=True,
                              collate_fn=data_collator,
                              batch_size=train_args.micro_batch_size,
                              num_workers=2
                              )
eval_dataloader = DataLoader(data["test"],
                             collate_fn=data_collator,
                             batch_size=train_args.micro_batch_size,
                             num_workers=2
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

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

# model = torch.compile(model)
logger.info(model)

# Optimizer
# Split weights in two groups, one with weight decay and the other not.
# no_decay = ["bias", "layer_norm.weight"]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,  # todo
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]
optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate)

num_training_steps = int(len(train_dataloader) * train_args.epoch / train_args.batch_size)
train_args.save_steps = num_training_steps // 2
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps * 0.1,
    num_training_steps=num_training_steps,
)

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

experiment_config = {}
experiment_config.update(dataclasses.asdict(train_args))
experiment_config.update(dataclasses.asdict(lora_args))
experiment_config['lora_target_modules'] = str(experiment_config['lora_target_modules'])  # todo: can not has list
accelerator.init_trackers("clm_no_trainer", experiment_config)

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataloader)}")
logger.info(f"  Num Epochs = {train_args.epoch}")
logger.info(f"  Instantaneous batch size per device = {train_args.micro_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {train_args.batch_size}")
logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {num_training_steps}")
# Only show the progress bar once on each machine.
progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0
mini_batch_loss = 0
epoch_ = int(train_args.epoch)
epoch_ = epoch_ + 1 if train_args.epoch > epoch_ else epoch_
for epoch in range(starting_epoch, epoch_):
    model.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            loss_ = loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        mini_batch_loss += loss_.item()
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
        else:
            continue  # for accelerator's gradient_accumulation

        lr = lr_scheduler.get_lr()[0]
        mini_batch_loss = mini_batch_loss / train_args.gradient_accumulation_steps
        logger.info(f"step:{completed_steps} train_loss:{mini_batch_loss} learning_rate:{lr}")
        accelerator.log(
            {
                "train_loss": mini_batch_loss,
                "learning_rate": lr,
            },
            step=completed_steps,
        )
        mini_batch_loss = 0

        if completed_steps % train_args.save_steps == 0:
            output_dir = f"step_{completed_steps}"
            output_dir = os.path.join(SAVE_PATH, output_dir)
            accelerator.save_state(output_dir)

        if completed_steps % train_args.eval_steps == 0:
            model.eval()
            losses = []
            for _, batch_ in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch_)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(train_args.micro_batch_size)))
            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            logger.info(f"step {completed_steps}: eval_loss: {eval_loss}")
            accelerator.log(
                {
                    "eval_loss": eval_loss,
                },
                step=completed_steps,
            )
            model.train()

        if completed_steps >= num_training_steps:
            break

accelerator.end_training()

accelerator.wait_for_everyone()
model.state_dict = old_state_dict
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    SAVE_PATH, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=True,
)
if accelerator.is_main_process:
    tokenizer.save_pretrained(SAVE_PATH)

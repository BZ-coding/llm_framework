"""
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 cuda-toolkit cudnn -c pytorch -c nvidia
conda install cuda-nvcc=12.1 -c pytorch -c nvidia
conda install numpy=1.23.5 fsspec=2023.9.2 sentencepiece protobuf transformers peft
![make `from torch._six import inf` to `from torch import inf`](https://github.com/microsoft/DeepSpeed/issues/2845)

<https://huggingface.co/docs/accelerate/main/en/usage_guides/megatron_lm#prerequisites>

# install apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--global-option=--cpp_ext" --config-settings "--global-option=--cuda_ext" ./
<https://github.com/InternLM/InternLM/issues/87>

# run accelerate with megatron
accelerate launch --config_file accelerate_megatron_config.yaml train_with_megatron.py
"""
import dataclasses
import os
import logging
import math

import torch
from accelerate.utils import MegatronLMOptimizerWrapper, MegatronLMSchedulerWrapper, MegatronLMDummyDataLoader
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger as accelerate_get_logger
from datasets import load_dataset
from transformers import get_scheduler, AutoModelForCausalLM, AutoConfig
from tqdm.auto import tqdm

from utils.args import TrainArgs, LoraArgs
from utils.data import get_generate_and_tokenize_prompt_fn
from utils.tokenizer import get_tokenizer
import utils.megatron_model_config
from tools.log import get_logger

BASE_MODEL = '/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma'
DATA_PATH = '/mnt/nfs/zsd_server/data/origin/alpaca_data_cleaned_archive.json'
SAVE_PATH = '/mnt/nfs/zsd_server/models/my/llama-7b_save'
is_megatron_dataset = False
project_name = 'clm_no_trainer'

train_args = TrainArgs(
    epoch=2,  # 0.008 for test
    gradient_accumulation_steps=8,
    micro_batch_size=4,
    max_length=1024,
    eval_steps=0,
    dtype="bf16",  # TODO: 修改环境变量？还是以配置文件里的为准？
)

lora_args = LoraArgs(
    lora_target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)

megatron_train_args = {
    "other_megatron_args": {
        "tokenizer_model": os.path.join(BASE_MODEL, "tokenizer.model"),
        "finetune": False,
        # "lora_target_modules": lora_args.lora_target_modules,
        "recompute_granularity": "full",
        "recompute_method": "block",
        "recompute_num_layers": 32,  # model's config.json
        "optimizer": "adam",
        "lr": train_args.learning_rate,
    }
}
from utils.megatron_gpt import train_valid_test_datasets_provider, get_batch as megatron_gpt_get_batch, \
    model_provider as megatron_gpt_model_provider
train_valid_test_datasets_provider.is_distributed = True
megatron_train_args_with_megatron_dataset = {
    "custom_megatron_datasets_provider_function": train_valid_test_datasets_provider,
    "custom_get_batch_function": megatron_gpt_get_batch,  # 需要注意megatron的get_batch只能用于megatron的数据
    "custom_model_provider_function": megatron_gpt_model_provider,
}
if is_megatron_dataset:
    megatron_train_args.update(megatron_train_args_with_megatron_dataset)

megatron_dataloader_config = {
    "data_path": [DATA_PATH],
    "splits_string": '949,50,1',
    "seq_length": train_args.max_length,
    "micro_batch_size": train_args.micro_batch_size,
}

transformer_dataloader_config = {
    "return_tensors": "pt",
    "padding": True,
    "pad_to_multiple_of": 8,
    # "pad_to_multiple_of": train_args.max_length,
}

if os.path.exists(SAVE_PATH):
    os.system(f"rm -rf {SAVE_PATH}")  # 在nfs上shutil.rmtree会报正忙、非空
os.makedirs(SAVE_PATH, exist_ok=True)

accelerate_kwargs = {
    "log_with": train_args.report_to,
    "project_dir": SAVE_PATH,
}
if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true":
    # TODO: model_provider等
    megatron_lm_plugin = train_args.get_megatron_train_args(
        **megatron_train_args
    )
    accelerate_kwargs["megatron_lm_plugin"] = megatron_lm_plugin
else:
    accelerate_kwargs["gradient_accumulation_steps"] = train_args.gradient_accumulation_steps
accelerator = Accelerator(**accelerate_kwargs)

log_level = logging.WARNING
if accelerator.is_local_main_process:
    log_level = logging.INFO
logger = accelerate_get_logger(__name__)
_ = get_logger(log_level=log_level, logger_log_level=log_level, logger=logger.logger)

logger.info(accelerator.state)
accelerator.wait_for_everyone()

if is_megatron_dataset:
    megatron_dataloader = MegatronLMDummyDataLoader(**megatron_dataloader_config)
    train_dataloader = megatron_dataloader
    eval_dataloader = megatron_dataloader
    accelerator.state.megatron_lm_plugin.megatron_dataset_flag = True
else:
    tokenizer = get_tokenizer(tokenizer_path=BASE_MODEL, use_fast=False, logger=logger)
    with accelerator.main_process_first():
        logger.info("Start handle dataset.", main_process_only=False)
        data = load_dataset("json", data_files=DATA_PATH)

        generate_and_tokenize_prompt = get_generate_and_tokenize_prompt_fn(tokenizer=tokenizer,
                                                                           max_length=train_args.max_length)

        data = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)
        data["train"] = data["train"].map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names,
                                          num_proc=2)
        data["test"] = data["test"].map(generate_and_tokenize_prompt, remove_columns=data["test"].column_names, num_proc=2)
        logger.info(data)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        **transformer_dataloader_config
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

# TODO: batch_size如何兼容fsdp、deepspeed、megatron获取
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    if is_megatron_dataset:
        dataloader_len = 1000
    else:
        dataloader_len = len(train_dataloader)
    train_args.dp = accelerator.num_processes // (accelerator.state.megatron_lm_plugin.tp_degree * accelerator.state.megatron_lm_plugin.pp_degree)
    train_args.batch_size = train_args.micro_batch_size * train_args.gradient_accumulation_steps * train_args.dp
    num_training_steps = math.ceil(
        dataloader_len * train_args.micro_batch_size * train_args.epoch / train_args.batch_size)
    accelerator.state.megatron_lm_plugin.megatron_lm_default_args["train_iters"] = num_training_steps
else:  # TODO: right? num_training_steps如何兼容fsdp、deepspeed、megatron获取
    train_args.dp = accelerator.num_processes
    train_args.batch_size = train_args.micro_batch_size * train_args.gradient_accumulation_steps * train_args.dp
    num_training_steps = math.ceil(
        len(train_dataloader) * train_args.micro_batch_size * train_args.epoch / (train_args.batch_size / train_args.gradient_accumulation_steps))

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    # 这里的model对megatron只提供config，所以要开lora需要在上面传参
    model_config = AutoConfig.from_pretrained(BASE_MODEL)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config)
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # load_in_8bit=True,
        torch_dtype=train_args.get_torch_dtype(),
        # device_map="cuda",  # "auto"
        # device_map="cpu"
    )
    model.gradient_checkpointing_enable()

model.config.use_cache = False

# model = torch.compile(model)

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    # TODO：自定义model（model_preovider）、optimizer和lr_scheduler
    # TODO：可否使用megatron格式的模型，即没有hf的config文件？
    # 需要修改accelerator的_prepare_megatron_lm里的逻辑
    if is_megatron_dataset:
        model, train_dataloader, eval_dataloader, _ = accelerator.prepare(
            model, train_dataloader, eval_dataloader, megatron_dataloader
        )
    else:
        model, train_dataloader, eval_dataloader = accelerator.prepare(
            model, train_dataloader, eval_dataloader
        )
    optimizer = MegatronLMOptimizerWrapper(model.optimizer)
    lr_scheduler = MegatronLMSchedulerWrapper(model.scheduler, model.optimizer)

    accelerator.load_state(SAVE_PATH)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.1 * train_args.gradient_accumulation_steps,
        num_training_steps=num_training_steps * train_args.gradient_accumulation_steps,
    )
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

experiment_config = {}
experiment_config.update(dataclasses.asdict(train_args))
experiment_config.update(dataclasses.asdict(lora_args))
accelerator.init_trackers(project_name, experiment_config)

logger.info("***** Running training *****")
logger.info(f"  Num examples = {num_training_steps * train_args.batch_size}")
logger.info(f"  Num Epochs = {train_args.epoch}")
logger.info(f"  Instantaneous batch size per device = {train_args.micro_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {train_args.batch_size}")
logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
logger.info(f"  Total steps = {num_training_steps}")
# Only show the progress bar once on each machine.
progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0
mini_batch_loss = 0


model.train()
while completed_steps < num_training_steps:
    # TODO: skip resume steps
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
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
        lr = lr_scheduler.get_lr()
        if accelerator.distributed_type != DistributedType.MEGATRON_LM:
            mini_batch_loss = mini_batch_loss / train_args.gradient_accumulation_steps
        logger.info(f"step:{completed_steps}\tlm loss:{mini_batch_loss}\tlearning rate:{lr}")
        accelerator.log(
            {
                "train_loss": mini_batch_loss,
                "learning_rate": lr,
            },
            step=completed_steps,
        )
        mini_batch_loss = 0

        if train_args.save_steps and completed_steps % train_args.save_steps == 0:
            accelerator.save_state(SAVE_PATH)

        if train_args.eval_steps and completed_steps % train_args.eval_steps == 0:
            model.eval()
            losses = []
            for _, batch_ in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch_)
                loss = outputs.loss
                if accelerator.distributed_type == DistributedType.MEGATRON_LM:
                    losses.append(loss)
                else:
                    losses.append(accelerator.gather_for_metrics(loss.repeat(train_args.batch_size)))
            if accelerator.distributed_type == DistributedType.MEGATRON_LM:
                losses = torch.tensor(losses)
            else:
                losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            logger.info(f"step:{completed_steps}\teval_loss: {eval_loss}")
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

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    accelerator.save_state(SAVE_PATH)
else:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        SAVE_PATH, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
if accelerator.is_main_process and not is_megatron_dataset:
    tokenizer.save_pretrained(SAVE_PATH)

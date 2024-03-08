from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union, Optional

import torch
from accelerate.utils import MegatronLMPlugin


class Dtype(Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass
class TrainArgs:
    max_length: int = 512
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    dp: int = 1
    learning_rate: int = 2e-5
    epoch: float = 2.0
    report_to: Optional[List[str]] = field(default_factory=list)
    save_steps: int = 0
    eval_steps: int = 0
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    dtype: Dtype = Dtype.BF16
    optim: str = "adamw_torch"
    output_dir: str = None
    logging_steps: int = 1
    logging_dir: str = None

    def __post_init__(self):
        self.batch_size = self.micro_batch_size * self.gradient_accumulation_steps * self.dp

        if type(self.dtype) is str:
            self.dtype = Dtype(self.dtype)
        self.fp32 = False
        self.fp16 = False
        self.bf16 = False
        if self.dtype == Dtype.BF16:
            self.bf16 = True
        elif self.dtype == Dtype.FP16:
            self.fp16 = True
        elif self.dtype == Dtype.FP32:
            self.fp32 = True

    def get_training_args(self):
        _training_args = {
            "per_device_train_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.epoch,
            "learning_rate": self.learning_rate,
            "output_dir": self.output_dir,
        }
        if self.warmup_ratio > 0.0:
            _training_args["warmup_ratio"] = self.warmup_ratio
        elif self.warmup_steps > 0:
            _training_args["warmup_steps"] = self.warmup_steps
        if self.bf16:
            _training_args["bf16"] = True
        elif self.fp16:
            _training_args["fp16"] = True
        if self.optim:
            _training_args["optim"] = self.optim
        if self.save_steps > 0:
            _training_args["save_steps"] = self.save_steps
            _training_args["save_strategy"] = "steps"
        if self.eval_steps > 0:
            _training_args["eval_steps"] = self.eval_steps
            _training_args["evaluation_strategy"] = "steps"
        if self.report_to:
            _training_args["report_to"] = self.report_to
        if self.logging_steps > 0 and self.logging_dir:
            _training_args["logging_steps"] = self.logging_steps
            _training_args["logging_dir"] = self.logging_dir

        return _training_args

    def get_megatron_train_args(self, other_megatron_args=None, **kwargs):
        megatron_lm_plugin = MegatronLMPlugin(
            seq_length=self.max_length,
            num_micro_batches=self.gradient_accumulation_steps,
            # tensorboard_dir=train_args,
            other_megatron_args=other_megatron_args,
            **kwargs
        )
        return megatron_lm_plugin

    def get_torch_dtype(self):
        if self.dtype == Dtype.BF16:
            return torch.bfloat16
        elif self.dtype == Dtype.FP16:
            return torch.float16
        elif self.dtype == Dtype.FP32:
            return torch.float32


@dataclass
class LoraArgs:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = field(default=None)

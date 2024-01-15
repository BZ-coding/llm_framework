import dataclasses
import functools
from dataclasses import dataclass, field
from typing import List, Union, Optional

from accelerate.utils import MegatronLMPlugin


@dataclass
class TrainArgs:
    max_length: int = 512
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    world_size: int = 1
    batch_size: int = micro_batch_size * gradient_accumulation_steps * world_size
    learning_rate: int = 2e-5
    epoch: float = 2.0
    log_file: str = '../train.log'
    report_to: Optional[List[str]] = field(default_factory=list)
    save_steps: int = 0
    eval_steps: int = 0
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_torch"
    output_dir: str = None
    logging_steps: int = 1
    logging_dir: str = None

    def __post_init__(self):
        assert self.fp16 != self.bf16

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

    def get_megatron_train_args(self, **kwargs):
        megatron_lm_plugin = MegatronLMPlugin(
            seq_length=self.max_length,
            num_micro_batches=self.gradient_accumulation_steps,
            # tensorboard_dir=train_args
            **kwargs
        )
        megatron_lm_plugin.set_training_args(
            micro_batch_size=self.micro_batch_size,
            dp_degree=self.world_size)
        return megatron_lm_plugin


@functools.lru_cache()
def get_train_args(**kwargs):
    return TrainArgs(**kwargs)


@dataclass
class LoraArgs:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = field(default=None)


@functools.lru_cache()
def get_lora_args(**kwargs):
    return LoraArgs(**kwargs)

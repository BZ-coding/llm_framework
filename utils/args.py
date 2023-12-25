import dataclasses
import functools
from dataclasses import dataclass, field
from typing import List, Union

from accelerate.utils import MegatronLMPlugin


@dataclass
class TrainArgs:
    max_length: int = 512
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    world_size: int = 1
    batch_size: int = micro_batch_size * gradient_accumulation_steps * world_size
    learning_rate: int = 2e-5
    epoch: float = 2.0
    log_file: str = '../train.log'
    report_to: str = 'tensorboard'
    save_steps: int = 50
    eval_steps: int = 50


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


# @functools.lru_cache()  # unhashable type: 'TrainArgs'
def get_megatron_train_args(train_args: TrainArgs, **kwargs):
    megatron_lm_plugin = MegatronLMPlugin(
        seq_length=train_args.max_length,
        num_micro_batches=train_args.gradient_accumulation_steps,
        # tensorboard_dir=train_args
        **kwargs
    )
    megatron_lm_plugin.set_training_args(micro_batch_size=train_args.micro_batch_size,
                                         dp_degree=train_args.world_size)
    return megatron_lm_plugin

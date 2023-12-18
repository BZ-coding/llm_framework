import dataclasses
import functools
from dataclasses import dataclass, field
from typing import List, Union


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
    report_to: str = 'tensorboard'  # todo: start tensorboard in python
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

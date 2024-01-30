#accelerate config --config_file accelerate_deepspeed_config.yaml

deepspeed train_with_trainer.py | tee train.log
#accelerate launch --config_file accelerate_deepspeed_config.yaml train_with_trainer.py | tee train.log

#accelerate launch --config_file accelerate_megatron_config.yaml train_with_megatron.py | tee train.log

#python train_with_trainer.py | tee train.log

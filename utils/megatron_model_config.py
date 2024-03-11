from accelerate.utils import add_model_config_to_megatron_parser


@add_model_config_to_megatron_parser('llama')
def parse_llama_config(megatron_lm_plugin, model, batch_data):
    model_type_name = "gpt"
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    max_position_embeddings = model.config.max_sequence_length
    orig_vocab_size = model.config.vocab_size
    pretraining_flag = True
    if megatron_lm_plugin.seq_length is None:
        if megatron_lm_plugin.decoder_seq_length is not None:
            megatron_lm_plugin.seq_length = megatron_lm_plugin.decoder_seq_length
        elif batch_data is not None:
            megatron_lm_plugin.seq_length = batch_data["input_ids"].shape[1]
        else:
            megatron_lm_plugin.seq_length = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["seq_length"] = megatron_lm_plugin.seq_length
    megatron_lm_plugin.megatron_lm_default_args["return_logits"] = megatron_lm_plugin.return_logits
    megatron_lm_plugin.megatron_lm_default_args["tokenizer_type"] = "Llama2Tokenizer"
    megatron_lm_plugin.megatron_lm_default_args["model_type_name"] = model_type_name
    megatron_lm_plugin.megatron_lm_default_args["num_layers"] = num_layers
    megatron_lm_plugin.megatron_lm_default_args["hidden_size"] = hidden_size
    megatron_lm_plugin.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
    megatron_lm_plugin.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
    megatron_lm_plugin.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
    megatron_lm_plugin.megatron_lm_default_args["model_return_dict"] = model.config.return_dict

from transformers import AutoTokenizer


def get_tokenizer(tokenizer_path, pad_token_id=0, use_fast=True, logger=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)
    tokenizer.pad_token_id = pad_token_id  # 0, unk. we want this to be different from the eos token
    if logger:
        logger.info(f"tokenizer.pad_token_id:{tokenizer.pad_token_id} tokenizer.pad_token:{tokenizer.pad_token}")
        logger.info(f"tokenizer.bos_token_id:{tokenizer.bos_token_id} tokenizer.bos_token:{tokenizer.bos_token}")
        logger.info(f"tokenizer.eos_token_id:{tokenizer.eos_token_id} tokenizer.eos_token:{tokenizer.eos_token}")
    return tokenizer

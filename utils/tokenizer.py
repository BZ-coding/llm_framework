import logging

from transformers import AutoTokenizer


def get_tokenizer(tokenizer_path, pad_token_id=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = pad_token_id  # 0, unk. we want this to be different from the eos token
    logging.info(f"tokenizer.pad_token_id:{tokenizer.pad_token_id} tokenizer.pad_token:{tokenizer.pad_token}")
    logging.info(f"tokenizer.bos_token_id:{tokenizer.bos_token_id} tokenizer.bos_token:{tokenizer.bos_token}")
    logging.info(f"tokenizer.eos_token_id:{tokenizer.eos_token_id} tokenizer.eos_token:{tokenizer.eos_token}")
    return tokenizer

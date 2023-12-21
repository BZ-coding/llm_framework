def alpaca_prompt(data_point):
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
             "Write a response that appropriately completes the request.\n\n" \
             f"### Instruction:\n{data_point['instruction']}"

    if data_point['input']:
        prompt += f"""\n\n### Input:\n{data_point['input']}"""

    prompt += f"""\n\n### Response:\n"""

    return prompt, data_point['output']


def _prompt_to_token(tokenizer, prompt, response, max_length=512, add_eos_token=True):
    prompt_result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    response_result = tokenizer(
        response,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    result = {
        "input_ids": prompt_result["input_ids"] + response_result["input_ids"][1:],
        "attention_mask": prompt_result["attention_mask"] + response_result["attention_mask"][1:]
    }
    result["input_ids"] = result["input_ids"][:max_length]
    result["attention_mask"] = result["attention_mask"][:max_length]

    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < max_length and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    len_prompt = len(prompt_result["input_ids"])
    result["labels"][:len_prompt] = [-100] * len_prompt  # -100 will be automatically ignored by PyTorch loss functions
    result["labels"] = result["labels"][:max_length]

    return result


def get_generate_and_tokenize_prompt_fn(tokenizer, max_length=512):
    def generate_and_tokenize_prompt(data_point):
        prompt, response = alpaca_prompt(data_point)
        tokenized_full_prompt = _prompt_to_token(tokenizer=tokenizer,
                                                 prompt=prompt,
                                                 response=response,
                                                 max_length=max_length,
                                                 add_eos_token=True)
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


if __name__ == "__main__":
    from utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_path='/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma')
    generate_and_tokenize_prompt = get_generate_and_tokenize_prompt_fn(tokenizer=tokenizer,
                                                                       max_length=512)

    a = generate_and_tokenize_prompt({'instruction': 'hello', 'input': 'test', 'output': 'Hello.'})
    print(a)
    print(type(a['input_ids']))
    print(tokenizer.decode(a['input_ids']))

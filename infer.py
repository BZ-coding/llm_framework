import time

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer

# model_path = '/mnt/nfs/zsd_server/models/huggingface/chinese-alpaca-2-7b'
# model_path = '/mnt/nfs/zsd_server/models/huggingface/llama-2-7B'
# model_path = '/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma'
model_path = '/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3'
# lora_path = '/mnt/nfs/zsd_server/models/my/llama-7b_save/step_1608'
lora_path = '/mnt/nfs/zsd_server/models/my/llama-7b_save'

# instruction = "你好"
# instruction = "hello"
instruction = "Give me three healthy life tips."

print(f"LlamaTokenizer from_pretrained start...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# tokenizer.bos_token_id = 1  # <s>
# tokenizer.eos_token_id = 2  # </s>
print(f"LlamaForCausalLM from_pretrained start...")
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             device_map='auto',
                                             # device_map='cuda',
                                             )
# model = PeftModel.from_pretrained(model=model,
#                                   model_id=lora_path,
#                                   device_map='auto',
#                                   use_safetensors=True
#                                   )
# model = model.merge_and_unload()
for n, p in model.named_parameters():
    if p.device.type == "meta":
        print(f"{n} is on meta!")
# print(model)

model = torch.compile(model)

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
 ### Instruction:
 {instruction}
 ### Response:
 """
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Below is an instruction that describes a task. Write a response that appropriately completes the request.<|eot_id|>
<|start_header_id|>user<|end_header_id|>Instruction: {instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>Answer:
"""
print(f"tokenizer start...")
inputs = tokenizer([prompt.format(instruction=instruction)], return_tensors="pt").to(model.device)

generate_input = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask,
    "max_new_tokens": 512,
    # "do_sample": True,
    "do_sample": False,
    "top_k": 50,
    # "top_p": 0.95,
    # "temperature": 0.3,
    "repetition_penalty": 1.3,
    # "eos_token_id":tokenizer.eos_token_id,
    # "bos_token_id":tokenizer.bos_token_id,
    # "pad_token_id":tokenizer.pad_token_id
}

with torch.inference_mode():
    print("Start Stream")
    streamer = TextStreamer(tokenizer)
    t0 = time.time()
    _ = model.generate(**generate_input, streamer=streamer)
    t1 = time.time()
    print(f"Output generated in {(t1 - t0):.2f} seconds")

while True:
    with torch.inference_mode():
        instruction = input("\n\n\nYour instruction: ('q' will quit)\n")
        if instruction == "q":
            exit()
        inputs = tokenizer([prompt.format(instruction=instruction)],
                           return_tensors="pt").to(model.device)
        generate_input = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 512,
            "do_sample": False,
        }
        streamer = TextStreamer(tokenizer)
        t0 = time.time()
        _ = model.generate(**generate_input, streamer=streamer)
        t1 = time.time()
        print(f"Output generated in {(t1 - t0):.2f} seconds")

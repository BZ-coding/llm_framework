import time

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer

# model_path = '/mnt/nfs/zsd_server/models/huggingface/chinese-alpaca-2-7b'
# model_path = '/mnt/nfs/zsd_server/models/huggingface/llama-2-7B'
model_path = '/mnt/nfs/zsd_server/models/huggingface/llama-7b-hf_yahma'
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
model = PeftModel.from_pretrained(model=model,
                                  model_id=lora_path,
                                  device_map='auto',
                                  use_safetensors=True
                                  )
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
print(f"tokenizer start...")
inputs = tokenizer([prompt.format(instruction=instruction)], return_tensors="pt").to('cuda')
# inputs = tokenizer([instruction,], return_tensors="pt").to('cuda')
# print(inputs)
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

# model.eval()
with torch.inference_mode():
    print("Start")
    t0 = time.time()
    generated = model.generate(**generate_input)
    t1 = time.time()
    print(f"Output generated in {(t1 - t0):.2f} seconds")
    print(tokenizer.decode(generated[0]))

print("\n\n\n")

with torch.inference_mode():
    print("Start Stream")
    streamer = TextStreamer(tokenizer)
    t0 = time.time()
    _ = model.generate(**generate_input, streamer=streamer)
    t1 = time.time()
    print(f"Output generated in {(t1 - t0):.2f} seconds")

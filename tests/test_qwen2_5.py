import torch
import warnings
warnings.filterwarnings("ignore")

from clean_llm.models.qwen2_5 import Qwen2_5
from transformers import AutoTokenizer



if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_path = "huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
# model_path = "huggingface_models/Qwen/Qwen3-0.6B"       # head_dim 128 不太一样
model = Qwen2_5.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"[INFO] Load {model_path.split('/')[-1]} model on device {device}")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
input_ids = model_inputs["input_ids"]

generated_idx = model.generate(
    input_ids,
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id
)

response_ids = generated_idx[0][len(input_ids[0]):]
response = tokenizer.decode(response_ids, skip_special_tokens=True)

print("Prompt:")
print(prompt)
print("Response:")
print(response)

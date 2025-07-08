import os
import sys
import json
import torch
import pickle
import pathlib
import hydra
from omegaconf import DictConfig
from clean_llm.eval.eval_pretrain import evaluate
from clean_llm.models.qwen2_5 import Qwen2_5
from clean_llm.tokenizer.tokenizer import get_custom_tokenizer
from clean_llm.utils import _to_device_and_compile

@hydra.main(config_path="configs", config_name="evaluate_config")
def main(cfg: DictConfig):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_config, eval_config, tokenizer_config = cfg.model, cfg.eval, cfg.tokenizer
    
    tokenizer = get_custom_tokenizer(**tokenizer_config)
    # import pdb; pdb.set_trace()
    model_config.vocab_size = tokenizer.vocab_size
    model_config.eos_token_id = tokenizer.eos_token_id
    model = Qwen2_5.from_config(model_config)

    model, device = _to_device_and_compile(model)

    with open(os.path.join(eval_config.save_path, f"ckpt_iter{eval_config.iteration}.pt"), 'rb') as f:
        checkpoint = torch.load(f, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])


    # 生成与输出
    result_text = evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=eval_config.prompt,
        max_new_tokens=eval_config.max_new_tokens,
        temperature=eval_config.temperature,
        top_k=eval_config.top_k,
        eos_token_id=tokenizer.eos_token_id  # 视你的tokenizer设置而定
    )
    print("输入：", eval_config.prompt)
    print("生成结果：", result_text)

if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig
import torch
import warnings
warnings.filterwarnings("ignore")

from clean_llm.models.qwen2_5 import Qwen2_5
from clean_llm.models.cs336_lm import BasicsTransformerLM
from clean_llm.train.pretrain import train
from clean_llm.tokenizer.tokenizer import get_custom_tokenizer
from clean_llm.utils import _to_device_and_compile



@hydra.main(config_path="configs/", config_name="pretrain_qwen2_5")
def main(cfg: DictConfig):

    model_config, training_config, tokenizer_config = cfg.model, cfg.training, cfg.tokenizer

    if cfg.model_type == "qwen2_5":
        tokenizer = get_custom_tokenizer(**tokenizer_config)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.eos_token_id = tokenizer.eos_token_id
        model = Qwen2_5.from_config(model_config)
    elif cfg.model_type == "cs336_lm":
        model = BasicsTransformerLM(**model_config)

    model, device = _to_device_and_compile(model)

    train(model, training_config)

if __name__ == '__main__':
    main()

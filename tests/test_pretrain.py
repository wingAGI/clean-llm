import hydra
from omegaconf import DictConfig
import torch
import warnings
warnings.filterwarnings("ignore")

from clean_llm.models.qwen2_5 import Qwen2_5
from clean_llm.train.pretrain import train

@hydra.main(config_path="configs/", config_name="config")
def main(cfg: DictConfig):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # model_path = "/Users/hex/workspace2/wingAGI/clean-llm/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
    # model = Qwen2_5.from_pretrained(model_path).to(device)
    model_config, training_config = cfg.model, cfg.training
    model = Qwen2_5.from_config(model_config).to(device)

    # import pdb; pdb.set_trace()
    train(model, training_config)

if __name__ == '__main__':
    main()

import os
import pickle
import hydra

from omegaconf import DictConfig
# from clean_llm.tokenizer.train import run_train_bpe       # slow version
from clean_llm.tokenizer.train_fast import run_train_bpe    # fast version


@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    vocab, merges = run_train_bpe(
        input_path=cfg.input_path,
        vocab_size=cfg.vocab_size,
        special_tokens=cfg.special_tokens
    )

    os.makedirs(cfg.tokenizer_dir, exist_ok=True)
    with open(cfg.vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(cfg.merges_path, "wb") as f:
        pickle.dump(merges, f)

    # 统计最长token
    longest_token = max(vocab.values(), key=len)
    print("最长token:", longest_token, "长度:", len(longest_token))


if __name__ == "__main__":
    main()

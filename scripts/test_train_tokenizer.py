import os
import pickle
import hydra

from omegaconf import DictConfig
from clean_llm.tokenizer.train import run_train_bpe
from clean_llm.tokenizer.train_fast import run_train_bpe


@hydra.main(config_path="configs", config_name="train_tokenizer", version_base=None)
def main(cfg: DictConfig):

    # 用 hydra 方式获取绝对路径
    # ROOT_DIR = hydra.utils.get_original_cwd()
    # data_path = os.path.join(ROOT_DIR, cfg.data_dir, cfg.input_file)
    # tokenizer_dir = os.path.join(ROOT_DIR, cfg.tokenizer.save_dir)
    # vocab_path = os.path.join(tokenizer_dir, cfg.tokenizer.vocab_file)
    # merges_path = os.path.join(tokenizer_dir, cfg.tokenizer.merges_file)

    # 训练
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

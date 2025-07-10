import hydra
from omegaconf import DictConfig
from clean_llm.tokenizer.tokenizer import get_custom_tokenizer, encode_txt_as_array

@hydra.main(config_path="configs", config_name="train_tokenizer", version_base=None)
def main(cfg: DictConfig):
    tokenizer = get_custom_tokenizer(vocab_path=cfg.vocab_path, 
                                     merges_path=cfg.merges_path, 
                                     special_tokens=cfg.special_tokens)

    encode_txt_as_array(tokenizer, cfg.train_txt_path, cfg.train_dat_path)
    encode_txt_as_array(tokenizer, cfg.valid_txt_path, cfg.valid_dat_path)


if __name__ == "__main__":
    main()
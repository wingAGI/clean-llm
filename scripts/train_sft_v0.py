import os
import time
import torch
import hydra
import mlflow
from pathlib import Path
from typing import Dict, List
from omegaconf import DictConfig
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from clean_llm.train.sft import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
    run_parse_gsm8k_response,
    evaluate_gsm8k
)
from clean_llm.utils import _to_device_and_compile, log_params_from_omegaconf_dict



def save_checkpoint(model, tokenizer, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ 保存模型到 {save_dir}")


def train(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_prompt_strs: List[str],
    train_output_strs: List[str],
    test_prompt_strs: List[str],
    test_output_strs: List[str],
):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.csv_dir, exist_ok=True)
    # 支持梯度累积
    micro_bs = cfg.micro_batch_size
    grad_accum_steps = cfg.gradient_accumulation_steps
    global_batch_size = micro_bs * grad_accum_steps

    num_epochs = cfg.num_epochs
    max_steps = cfg.get("max_steps", None)  # 可选，按 step 训练
    eval_steps = cfg.eval_steps
    save_steps = cfg.save_steps
    project_name = cfg.get("mlflow_project", "sft-gsm8k")
    run_name = cfg.get("mlflow_run", f"run-{int(time.time())}")

    # ------------ MLflow ------------
    mlflow.set_experiment(project_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(cfg)

    # ------------ 数据 ------------
    train_inputs = run_tokenize_prompt_and_output(
        train_prompt_strs, train_output_strs, tokenizer
    )
    for k, v in train_inputs.items():
        train_inputs[k] = v.to(cfg.train_device)

    # ------------ 训练状态 ------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    step = 0
    epoch = 0
    model.train()

    # ------------ 训练循环 ------------
    total_samples = len(train_prompt_strs)
    samples_seen = 0
    pbar = tqdm(total=total_samples * num_epochs, desc="Training")

    while True:
        for i in range(0, total_samples, micro_bs):
            # 构造 micro-batch
            end = min(i + micro_bs, total_samples)
            batch_input_ids = train_inputs["input_ids"][i:end]
            batch_labels = train_inputs["labels"][i:end]
            batch_response_mask = train_inputs["response_mask"][i:end]

            # 前向 & 计算 loss
            res = run_get_response_log_probs(
                model,
                batch_input_ids,
                batch_labels,
                return_token_entropy=False,
            )
            loss, _ = run_sft_microbatch_train_step(
                res["log_probs"],
                batch_response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1,
            )

            samples_seen += end - i
            pbar.update(end - i)

            # 梯度累积
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # ------------ 日志 ------------
            mlflow.log_metric("loss", loss.item(), step=step)

            # ------------ 评估 ------------
            if (step + 1) % eval_steps == 0:
                eval_res = evaluate_gsm8k(model_id=cfg.model_path,
                            policy=model,
                            tokenizer=tokenizer,
                            prompt_strs=test_prompt_strs,
                            output_strs=test_output_strs,
                            save_path=os.path.join(cfg.csv_dir, f'step_{step}.csv'),
                            device=cfg.eval_device
                            )
                mlflow.log_metrics(eval_res, step=step)
                model.train()

            # ------------ 保存 ------------
            if (step + 1) % save_steps == 0:
                ckpt_dir = Path(cfg.checkpoint_dir) / f"checkpoint-{step}"
                save_checkpoint(model, tokenizer, ckpt_dir)

            step += 1

            # 可选按 step 终止
            if max_steps and step >= max_steps:
                break

        epoch += 1
        if epoch >= num_epochs or (max_steps and step >= max_steps):
            break

    pbar.close()

    # ------------ 训练结束 ------------
    final_ckpt_dir = Path(cfg.checkpoint_dir) / "final"
    save_checkpoint(model, tokenizer, final_ckpt_dir)
    mlflow.end_run()
    print("🎉 训练完成")


# ---------------- Hydra 入口 ----------------
@hydra.main(config_path="configs", config_name="sft_gsm8k", version_base=None)
def main(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.bfloat16, device_map=cfg.train_device
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, padding_side='left')
    # model, device = _to_device_and_compile(model, cfg.train_device)
    print(f"Load huggingface model from {cfg.model_path} on device {cfg.train_device}")

    dataset = load_dataset(cfg.dataset_path, "main")
    train_prompt_strs = [ex["question"] for ex in dataset["train"]]
    train_output_strs = [ex["answer"] for ex in dataset["train"]]
    test_prompt_strs = [ex["question"] for ex in dataset["test"]]
    test_output_strs = [ex["answer"] for ex in dataset["test"]]

    print(f"Train samples: {len(train_prompt_strs)}, Test samples: {len(test_prompt_strs)}")


    train(
        cfg,
        model,
        tokenizer,
        train_prompt_strs,
        train_output_strs,
        test_prompt_strs,
        test_output_strs,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from typing import Any, Callable, Literal, Dict, List
from torch import Tensor
from torch.utils.data import Dataset
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch
from transformers import PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    初始化 vLLM LLM（大语言模型），可选择设备及显存利用率，在推理时与训练策略分离。
    参考自 HuggingFace TRL 实现。
    
    Args:
        model_id (str): 模型标识
        device (str): 目标设备，如 'cuda:0'
        seed (int): 随机种子
        gpu_memory_utilization (float): 显存占用比例
        
    Returns:
        LLM: vLLM初始化好的对象
    """
    vllm_set_random_seed(seed)
    
    # Patch 1：让vllm假装“集群”只有1卡
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # Patch 2：跳过vllm内部显存剖析的某个断言
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return llm

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    把训练好的PyTorch模型参数装载进vLLM实例。
    参考自 HuggingFace TRL。
    
    Args:
        policy (PreTrainedModel): 已训练好的transformers模型
        llm (LLM): vLLM实例
        
    Returns:
        None
    """
    state_dict = policy.state_dict()
    # 下面这一行依赖vllm当前内部实现，如有变动请根据vllm源代码调整！
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())



def evaluate_gsm8k(
    model_id: str,
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt_strs: List[str],
    output_strs: List[str],
    save_path,
    device="cuda:0",
    seed=0,
    gpu_memory_utilization=0.5
):
    llm = init_vllm(model_id, device, seed, gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=tokenizer.eos_token)
    outputs = llm.generate(prompt_strs, sampling_params)
    # import pdb; pdb.set_trace()

    result_df = pd.DataFrame(columns=['Prompt', 'Generated_Text', 'Correct_Answer', 'Parsed_Answer', 'Parsed_Correct_Answer', 'Evaluation_Score', 'ParseFail'])
    correct, parse_fail_cnt = 0, 0
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        correct_answer = output_strs[i]
        
        parsed_answer = parse_gsm8k_qwen_response(generated_text)
        parsed_correct_answer = run_parse_gsm8k_response(correct_answer)
        
        parse_fail = parsed_answer == None
        evaluation_score = 1 if parsed_correct_answer == parsed_answer else 0
        if evaluation_score == 1:
            correct += 1
        if parse_fail:
            parse_fail_cnt += 1
        
        
        temp_df = pd.DataFrame({
            'Prompt': [prompt],
            'Generated_Text': [generated_text],
            'Correct_Answer': [correct_answer],
            'Parsed_Answer': [parsed_answer],
            'Parsed_Correct_Answer': [parsed_correct_answer],
            'Evaluation_Score': [evaluation_score],
            'ParseFail': [parse_fail]
        })
        
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    print(f"Parse fail {parse_fail_cnt}/{len(outputs)}")
    print(f"Correct {correct}/{len(outputs)} Accuracy is {round(correct/len(outputs)*100, 2)}%")

    result_df.to_csv(save_path)

    
    return {
        'parse_fail_rate': parse_fail_cnt / len(outputs),
        'correct_rate': correct / len(outputs)
    }



def parse_gsm8k_qwen_response(
    model_output: str,
) -> str | None:
    matches = re.findall(r'```output(.*?)```', model_output, re.DOTALL)
    if matches:
        res = matches[0].strip()
    
        return res
    
    return None


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    input_ids_list = []
    response_mask_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_enc = tokenizer(prompt, add_special_tokens=False)
        output_enc = tokenizer(output, add_special_tokens=False)
        full_input = prompt_enc['input_ids'] + output_enc['input_ids']
        response_mask = [0] * len(prompt_enc['input_ids']) + [1] * len(output_enc['input_ids'])
        input_ids_list.append(torch.tensor(full_input, dtype=torch.long))
        response_mask_list.append(torch.tensor(response_mask, dtype=torch.long))

    batch_size = len(input_ids_list)
    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_batch = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, response_mask_list)):
        seq_len = len(ids)
        input_ids_batch[i, :seq_len] = ids
        response_mask_batch[i, :seq_len] = mask

    return {
        "input_ids": input_ids_batch[:, :-1],               # (batch, max_len-1)
        "labels": input_ids_batch[:, 1:],                   # (batch, max_len-1)
        "response_mask": response_mask_batch[:, 1:]         # (batch, max_len-1)
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    # 1. Compute raw rewards for all responses
    raw_rewards = []
    format_rewards = []
    answer_rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, gt)
        raw_rewards.append(reward_dict["reward"])
        format_rewards.append(reward_dict["format_reward"])
        answer_rewards.append(reward_dict["answer_reward"])
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)

    # 2. Group normalization
    N = len(raw_rewards)
    assert N % group_size == 0, "Rollout batch size must be divisible by group_size"
    n_groups = N // group_size


    # 分组: [n_groups, group_size]
    group_rewards = raw_rewards.view(n_groups, group_size)
    group_means = group_rewards.mean(dim=1, keepdim=True)
    if normalize_by_std:
        group_stds = group_rewards.std(dim=1, keepdim=True)
        denom = group_stds + advantage_eps
    else:
        denom = 1.0

    # 归一化
    normalized_groups = (group_rewards - group_means) / denom       # [n_groups, group_size]
    normalized_rewards = normalized_groups.view(N)                  # 还原回(N,)

    # 3. Optional: Collect some statistics
    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()),
        "reward_max": float(raw_rewards.max()),
        "reward_min": float(raw_rewards.min()),
        "format_reward_mean": float(np.mean(format_rewards)),
        "answer_reward_mean": float(np.mean(answer_rewards)),
    }

    return normalized_rewards, raw_rewards, metadata



def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # Numerically stable computation of entropy
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    expected_logit = (probs * logits).sum(dim=-1)
    entropy = lse - expected_logit
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # Get logits from model
    outputs = model(input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Compute log-probabilities for each token in the labels
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # Gather log-probabilities at the correct label indices
    # Unsqueeze `labels` to match log_probs' last dim for gather
    # Result shape: (batch_size, seq_len)
    log_probs_for_labels = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {
        "log_probs": log_probs_for_labels
    }

    if return_token_entropy:
        token_entropy = run_compute_entropy(logits)  # (batch_size, seq_len)
        result["token_entropy"] = token_entropy

    return result


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    seq_length = policy_log_probs.shape[1]
    rewards_or_advantages = raw_rewards_or_advantages.expand(-1, seq_length)
    pg_loss = -policy_log_probs * rewards_or_advantages
    return pg_loss


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    seq_length = policy_log_probs.shape[1]
    advantages = advantages.expand(-1, seq_length)

    ratio = torch.exp(policy_log_probs - old_log_probs)                 # (batch_size, seq_length)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    lhs, rhs = ratio * advantages, clipped_ratio * advantages
    loss = -torch.min(lhs, rhs)

    metadata = {
        "clipped": (rhs < lhs).float(),
    }

    return loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    batch_size, seq_length = policy_log_probs.shape
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        assert raw_rewards.shape == (batch_size, 1), "raw_rewards must have shape (batch_size, 1)"
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        meta = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        assert advantages.shape == (batch_size, 1), "advantages must have shape (batch_size, 1)"
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        meta = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        assert old_log_probs.shape == (batch_size, seq_length), "old_log_probs must have shape (batch_size, seq_length)"
        loss, meta = run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss, meta



def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    n_tokens = mask.sum(dim=dim)
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / n_tokens

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size, seq_length = policy_log_probs.shape
    # Cross-entropy loss (neg log likelihood), per token and masked
    ce_loss = -policy_log_probs                                         # (batch_size, seq_length)
    # Sum over tokens and batch, only response tokens count
    loss_sum = run_masked_normalize(ce_loss, response_mask, normalize_constant=normalize_constant)

    loss = loss_sum / batch_size / gradient_accumulation_steps
    loss.backward()

    # For logging
    n_tokens = response_mask.sum()
    avg_token_ce = loss_sum / (n_tokens + 1e-8)
    metadata = {
        "loss_sum": loss_sum.detach(),
        "n_tokens": n_tokens.detach(),
        "avg_ce_per_token": avg_token_ce.detach(),
        "mean_log_prob": (policy_log_probs * response_mask).sum() / (n_tokens + 1e-8)
    }

    return loss.detach(), metadata



def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """

    # 1. 调用run_compute_policy_gradient_loss 得到逐token损失 (batch, seq)
    loss_per_token, meta = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 2. 用response_mask对loss按token聚合: (batch, )
    loss_per_example = run_masked_mean(loss_per_token, response_mask, dim=1)  # (batch, )

    # 3. 平均到batch: 标量
    loss = loss_per_example.mean()

    # 4. adjust for grad accumulation
    loss = loss / gradient_accumulation_steps

    # 5. backward
    loss.backward()

    # 6. meta 里可以增加点logging的信息
    meta = meta.copy()
    meta["microbatch_loss"] = loss.detach()
    meta["loss_per_example"] = loss_per_example.detach()

    return loss, meta

def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    sum_vals = masked_tensor.sum(dim=dim)
    normalized = sum_vals / normalize_constant
    return normalized


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError





def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # raise NotImplementedError
    import re
    numbers = re.findall(r'-?\d+\.?\d*', model_output)
    
    # 如果没有找到任何数字，返回 None
    if not numbers:
        return None
    
    # 返回最后一个数字
    return numbers[-1]


from __future__ import annotations

import os
import torch
import numpy as np
from typing import IO, BinaryIO
from collections.abc import Iterable



def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Filter parameters with gradients
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(parameters_with_grad) == 0:
        return
    
    # Calculate total L2 norm of all gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters_with_grad))
    
    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # Add small value to avoid division by zero
    
    # If total norm exceeds max_norm, scale down all gradients
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.mul_(clip_coef)

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super(AdamW, self).__init__(params, defaults)

        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Perform stepweight decay
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                    # Get parameter-specific state
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    step_size = group['lr']
                    eps = group['eps']

                    # Update state
                    state['step'] += 1
                    grad = p.grad.data

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = exp_avg_sq.sqrt().add_(eps)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

            return loss
    
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # Warm-up 阶段：线性增加学习率
        lr = (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine Annealing 阶段：余弦函数衰减
        t = it - warmup_iters
        T = cosine_cycle_iters - warmup_iters
        cos_value = np.cos(np.pi * t / T)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos_value)
    else:
        # Post-annealing 阶段：学习率保持最小值
        lr = min_learning_rate

    return lr

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            checkpoint = torch.load(f, weights_only=False)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']
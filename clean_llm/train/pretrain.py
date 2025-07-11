import os
import sys
import mlflow
import torch
import torch.nn.functional as F
import pathlib
import numpy as np
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .adapters import *



def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode="r")   # 单列token id序列
    return arr

def get_batch(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    ix = np.random.randint(0, N-context_length-1, size=(batch_size,))
    x = np.stack([memmap_arr[i:i+context_length] for i in ix])
    y = np.stack([memmap_arr[i+1:i+context_length+1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def memmap_val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)



def train(model, device, args):
    os.makedirs(args.save_path, exist_ok=True)

    # 2. 加载数据集
    train_data = get_memmap_dataset(args.train_data_path)
    val_data = get_memmap_dataset(args.val_data_path)

    # 3. 构建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. 恢复断点
    start_iter = 0
    if args.resume_checkpoint:
        print(f"Resuming from checkpoint {args.resume_checkpoint}")
        resume_ckpt_path = pathlib.Path(HydraConfig.get().runtime.output_dir) / f"{args.save_path}/ckpt_iter{args.resume_checkpoint}.pt"
        start_iter = run_load_checkpoint(resume_ckpt_path, model, optimizer)
        print(f"Resumed at iteration {start_iter} from path {resume_ckpt_path}")

    # 5. 训练loop

    pbar = tqdm(range(start_iter, args.train_steps), desc="Training", leave=False)
    for iteration in pbar:
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length)
        x, y = x.to(device), y.to(device)
        
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), args.clip_grad_norm)
        
        # 更新学习率
        lr = run_get_lr_cosine_schedule(
            iteration, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        pbar.set_postfix(loss=loss.item(), lr=lr)
        mlflow.log_metric("loss", loss.item(), step=iteration)
        mlflow.log_metric("lr", lr, step=iteration)

        # 验证
        if (iteration+1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in memmap_val_iterator(val_data, args.batch_size, args.context_length):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits, _ = model(x_val)
                    val_loss = F.cross_entropy(
                        val_logits.reshape(-1, val_logits.shape[-1]),
                        y_val.reshape(-1)
                    )
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.val_batches:
                        break
                val_loss_mean = np.mean(val_losses)
                mlflow.log_metric("val_loss", val_loss_mean, step=iteration)
                print(f"iter {iteration+1:05d}: VALID loss = {val_loss_mean:.4f}")
                

        # 保存
        if (iteration+1) % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{iteration+1}.pt")
            run_save_checkpoint(model, optimizer, iteration+1, ckpt_name)
            print(f"Checkpoint saved to {ckpt_name}")
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat(x, rep):
    """
    GQA 共享 KV 需要repeat.
    (B, N_G, T, H) -> (B, N, T, H)
    rep: N // N_G
    """
    B, N_G, T, H = x.shape
    x = x.unsqueeze(2)
    x = x.expand(-1, -1, rep, -1, -1).contiguous()
    x = x.view(B, N_G * rep, T, H)
    return x

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdims=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, embed_dim, immediate_dim, bias=False):
        super(SwiGLU, self).__init__()
        self.up_proj = nn.Linear(embed_dim, immediate_dim, bias=bias)
        self.down_proj = nn.Linear(immediate_dim, embed_dim, bias=bias)
        self.gate_proj = nn.Linear(embed_dim, immediate_dim, bias=bias)

    def forward(self, x):
        x, gate = self.up_proj(x), self.gate_proj(x)
        x = F.silu(gate) * x
        x = self.down_proj(x)
        
        return x


def _compute_rope_params(config):
    base = config.rope_theta
    if 'Qwen' in config.architectures[0]:
        rope_dim = config.hidden_size // config.num_attention_heads
    elif 'Deepseek' in config.architectures[0]:
        rope_dim = config.qk_rope_head_dim

    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2) / rope_dim))    # (dim // 2)
    T = config.max_position_embeddings
    position_ids_expanded = torch.arange(0, T).reshape(1, T)                # (1, T)
    inv_freq_expanded = inv_freq.reshape(-1, 1)                             # (dim // 2, 1)

    # (dim // 2, T) -- transpose --> (T, dim // 2)
    freqs = (inv_freq_expanded @ position_ids_expanded.float()).transpose(0, 1)             

    emb = torch.cat((freqs, freqs), dim=-1)                                 # (T, dim)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        base, head_dim = config.rope_theta, config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)                    # (dim // 2)

    @torch.no_grad()
    def forward(self, x, position_ids):
        B, T = position_ids.shape
        inv_freq_expanded = self.inv_freq[None, :, None].expand(B, -1, 1)               # (B, dim // 2, 1)
        position_ids_expanded = position_ids[:, None, :].float()                        # (B, 1, T)

        # (B, dim // 2, T) -- transpose --> (B, T, dim // 2)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)             

        emb = torch.cat((freqs, freqs), dim=-1)                                         # (B, T, dim)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


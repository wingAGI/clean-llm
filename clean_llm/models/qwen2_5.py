import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from .basics import RMSNorm, SwiGLU, apply_rotary_pos_emb, _compute_rope_params, repeat



class CasualGroupQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_group, proj_bias=False):
        super(CasualGroupQueryAttention, self).__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_group = n_group
        self.hs = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.kv_proj = nn.Linear(n_embd, 2 * self.n_group * self.hs)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=proj_bias)

    def forward(self, x, cos=None, sin=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hs).transpose(1, 2)  # (B, N, T, H)
        k, v = self.kv_proj(x).chunk(2, dim=-1)                              # (B, T, N_G * H)
        k = k.view(B, T, self.n_group, self.hs).transpose(1, 2)         # (B, N_G, T, H)
        v = v.view(B, T, self.n_group, self.hs).transpose(1, 2)
        
        k = repeat(k, self.n_head // self.n_group)
        v = repeat(v, self.n_head // self.n_group)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = q @ k.transpose(-1,-2) / math.sqrt(self.hs)
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(attn.device)
        attn = attn.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)                                    # (B, N, T, T)

        o = attn @ v                                                      # (B, N, T, H)
        o = o.transpose(1,2).contiguous().view(B, T, D)
        o = self.o_proj(o)   
        
        return o


class Block(nn.Module):

    def __init__(self, n_embd, n_head, n_group, immediate_dim):
        super(Block, self).__init__()
        self.attn = CasualGroupQueryAttention(n_embd, n_head, n_group)
        self.ffn = SwiGLU(n_embd, immediate_dim)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x, cos=None, sin=None):       
        x = x + self.attn(self.ln1(x), cos, sin)        # pre-norm
        x = x + self.ffn(self.ln2(x))

        return x


class Qwen2_5(nn.Module):

    def __init__(self, config_dict):
        super(Qwen2_5, self).__init__()
        config = SimpleNamespace(**config_dict)
        self.block_size = config.max_position_embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(n_embd=config.hidden_size, n_head=config.num_attention_heads, n_group=config.num_key_value_heads, immediate_dim=config.intermediate_size) for _ in range(config.num_hidden_layers)]
        )
        self.last_norm = RMSNorm(embed_dim=config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.lm_head.weight

        cos_cached, sin_cached = _compute_rope_params(config)
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=1.0, top_k=50):
        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_token_id is not None:
                if idx_next.item() == eos_token_id:
                    break

        return idx

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)                # (B, T, D)
        cos = self.cos_cached[:T, :].unsqueeze(0).expand(B, -1, -1)
        sin = self.sin_cached[:T, :].unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            x = layer(x, cos, sin)
        
        x = self.last_norm(x)
        logits = self.lm_head(x)

        return logits, None



    @classmethod
    def from_pretrained(cls, model_path):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        model = Qwen2_5(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        sd_hf = model_hf.state_dict()

        key_map = {'embed_tokens': 'embed_tokens', 'attn': 'self_attn', 'q_proj': 'q_proj', 'o_proj': 'o_proj', 'ffn': 'mlp', 
                    'ln1': 'input_layernorm', 'ln2': 'post_attention_layernorm', 'last_norm': 'norm'}

        def to_hf_key(key):
            components = key.split('.')
            for i, c in enumerate(components):
                if c in key_map.keys():
                    components[i] = key_map[c]

            if not key == 'lm_head.weight':
                key = 'model.' + '.'.join(components)
            
            return key

        for key in sd_keys:
            if key in ['cos_cached', 'sin_cached']:
                continue
            hf_key = to_hf_key(key)
            if 'kv_proj' in hf_key:
                hf_key_k, hf_key_v = hf_key.replace('kv_proj', 'k_proj'), hf_key.replace('kv_proj', 'v_proj')
                sd[key].copy_(torch.concat((sd_hf[hf_key_k], sd_hf[hf_key_v]), dim=0))
            else:
                # print("=" * 20)
                # print(key, hf_key)
                # print(sd[key].shape, sd_hf[hf_key].shape)
                sd[key].copy_(sd_hf[hf_key])
        
        return model
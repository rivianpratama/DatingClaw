import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache"
os.environ["TORCH_HOME"] = r"D:\hf_cache"

from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModel.from_pretrained(
    "facebook/dinov2-with-registers-base",
    cache_dir=r"D:\hf_cache",
    torch_dtype=torch.float16
).to(device).eval()

x = torch.randn(2, 3, 378, 378, dtype=torch.float16, device=device)

with torch.no_grad():
    hidden = model.embeddings(x)
    print(f"Hidden after embeddings: {hidden.shape}")
    for i, layer in enumerate(model.encoder.layer[:-1]):
        out = layer(hidden)
        hidden = out[0] if isinstance(out, tuple) else out
    print(f"Hidden before last layer: {hidden.shape}")
    
    last_layer = model.encoder.layer[-1]
    normed_in = last_layer.norm1(hidden)
    print(f"normed_in shape: {normed_in.shape}")
    
    B, N, C = normed_in.shape
    num_heads = last_layer.attention.attention.num_attention_heads
    head_dim = C // num_heads
    print(f"B={B}, N={N}, C={C}, num_heads={num_heads}, head_dim={head_dim}")
    
    q_cls = last_layer.attention.attention.query(normed_in[:, :1, :])
    k_all = last_layer.attention.attention.key(normed_in)
    
    q_cls = q_cls.view(B, 1, num_heads, head_dim).transpose(1, 2)
    k_all = k_all.view(B, N, num_heads, head_dim).transpose(1, 2)
    
    attn_logits = torch.matmul(q_cls, k_all.transpose(-1, -2)) / (head_dim ** 0.5)
    attn_probs = F.softmax(attn_logits.float(), dim=-1).squeeze(2)
    print(f"attn_probs shape: {attn_probs.shape}")
    
    a_spatial = attn_probs[:, :, 5:].mean(dim=1)
    w_patch = a_spatial / (a_spatial.sum(dim=-1, keepdim=True) + 1e-8)
    print(f"w_patch shape: {w_patch.shape}")
    
    last_out = last_layer(hidden)
    last_hidden = last_out[0] if isinstance(last_out, tuple) else last_out
    out_hidden = model.layernorm(last_hidden).float()
    x_cls = F.normalize(out_hidden[:, 0, :], p=2, dim=-1)
    p_patches = out_hidden[:, 5:, :]
    p_fg = torch.bmm(w_patch.unsqueeze(1), p_patches).squeeze(1)
    p_fg = F.normalize(p_fg, p=2, dim=-1)
    x_dino = F.normalize(torch.cat([x_cls, p_fg], dim=-1), p=2, dim=-1)
    print(f"x_dino shape: {x_dino.shape}, norm: {torch.norm(x_dino, dim=-1)}")
    print("SUCCESSFULLY VERIFIED DINOV2 EXTRACTION!")

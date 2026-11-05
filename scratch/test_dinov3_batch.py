import time
import torch
import torch.nn.functional as F
from transformers import AutoModel

path = r"C:\Users\Rivian\dinov3-vitb16-pretrain-lvd1689m"
print(f"Loading DINOv3 from {path}...")
model = AutoModel.from_pretrained(path).to("cuda").eval()

B = 32
x = torch.randn(B, 3, 384, 384, device="cuda")
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
x = (x - mean) / std

print(f"Running warm-up batch of {B} images (384x384)...")
t0 = time.time()
with torch.no_grad():
    out = model(x).last_hidden_state
    cls_tok = F.normalize(out[:, 0, :], p=2, dim=-1)
    patches = out[:, 5:, :]
    p_norm = F.normalize(patches, p=2, dim=-1)
    w = F.softmax(torch.bmm(p_norm, cls_tok.unsqueeze(-1)).squeeze(-1) / 0.15, dim=-1).unsqueeze(1)
    fg = F.normalize(torch.bmm(w, p_norm).squeeze(1), p=2, dim=-1)
    d1536 = F.normalize(torch.cat([cls_tok, fg], dim=-1), p=2, dim=-1)

torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"Success! {B} images took {elapsed:.3f}s ({B/elapsed:.1f} img/s) | Output shape: {d1536.shape}")
print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

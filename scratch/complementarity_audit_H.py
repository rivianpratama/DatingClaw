"""Read-only complementarity audit: base vs LoRA on F5 H (never T). Fixed 50/50 rank blend, no tuning."""
import json, numpy as np
from pathlib import Path
from scipy.stats import rankdata
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.provenance import deterministic_topk
R=Path(r"D:\bumbleclaw_v6r1\results")
snap=load_snapshot(Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935"))
m=snap["samples"].set_index(snap["samples"]["sample_id"].astype(str))
def ld(p):
    z=np.load(p,allow_pickle=True); return dict(zip(z["sample_id"].astype(str),z["logit"].astype(float)))
b={**ld(R/"e01r_dev4_base/predictions/F5_C_s0.npz"),**ld(R/"e01r_dev4_base/predictions/F5_P_s0.npz")}; l=ld(R/"e05r_lora_f5_long_r2/predictions/F5_H_s0.npz")
ids=np.array(sorted(set(b)&set(l))); y=m.loc[ids,"y"].to_numpy(); old=m.loc[ids,"old_score"].to_numpy(float)
sb=np.array([b[i] for i in ids]); sl=np.array([l[i] for i in ids]); K=int(np.floor(.208*len(ids)))
def sel(s): z=np.zeros(len(ids),bool); z[deterministic_topk(s,ids,K)]=True; return z
def stats(z):
    fp=z&(y==0); return dict(hits=int((z&(y==1)).sum()),severe=int((fp&(old<50)).sum()))
B,L=sel(sb),sel(sl); blend=sel(rankdata(sb)+rankdata(sl))
pos=y==1
out=dict(N=len(ids),K=K,n_pos=int(pos.sum()),base=stats(B),lora=stats(L),blend_50_50=stats(blend),
 both_hit=int((B&L&pos).sum()),base_only_hit=int((B&~L&pos).sum()),lora_only_hit=int((L&~B&pos).sum()),
 union_hit_ceiling=int(((B|L)&pos).sum()),
 base_only_severe=int((B&~L&~pos&(old<50)).sum()),lora_only_severe=int((L&~B&~pos&(old<50)).sum()),
 score_corr=float(np.corrcoef(rankdata(sb),rankdata(sl))[0,1]))
(R/"complementarity_audit_H.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

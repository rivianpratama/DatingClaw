"""Parity (training path: legacy boxes + stored side features) vs runtime path, plus latency. No labels used."""
import json, time, numpy as np, torch
from pathlib import Path
from bumbleclaw_v6.data.load import load_snapshot, FeatureBlock
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.features.preprocess import load_bgr
from bumbleclaw_v6.finetune.lora_siglip import load_bboxes
from bumbleclaw_v6.runtime.lora_scorer import LoRAEnsembleScorer, SIDE_KEYS
SNAP=Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
REL=Path(r"D:\bumbleclaw_v6r1\releases\v6r3_lora_photo_ens3")
meta=load_snapshot(SNAP)["samples"]; paths=meta["path"].astype(str).tolist()
bank=FeatureStoreBank(SNAP, Path(r"D:\bumbleclaw_v6r1\features\photo_v1_full_r2"), get_recipe("photo_v1"))
blocks=bank.blocks(SIDE_KEYS); boxes=load_bboxes()
fv=blocks["fv"].values[:,0]
rng=np.random.default_rng(0)
ok=np.flatnonzero(fv>0.5); nf=np.flatnonzero(fv<0.5)
rows=np.concatenate([rng.choice(ok,24,replace=False), rng.choice(nf,6,replace=False)])
sc=LoRAEnsembleScorer(REL)
rec=[]
for r in rows:
    side={k: FeatureBlock(k, blocks[k].values[r:r+1], blocks[k].mask[r:r+1], blocks[k].status[r:r+1]) for k in SIDE_KEYS}
    off=float(np.mean(sc.logits(load_bgr(paths[r]), boxes[r], side)))
    rt=sc.decide(paths[r])
    rec.append(dict(row=int(r), face=bool(fv[r]>0.5), offline=off, runtime=rt.get("logit"), reason=rt.get("reason")))
d=np.array([abs(x["offline"]-x["runtime"]) for x in rec if x["runtime"] is not None])
okd=np.array([abs(x["offline"]-x["runtime"]) for x in rec if x["runtime"] is not None and x["face"]])
lat=[]
for r in rng.choice(len(paths),100,replace=False):
    torch.cuda.synchronize(); t=time.perf_counter(); sc.decide(paths[r]); torch.cuda.synchronize(); lat.append((time.perf_counter()-t)*1e3)
lat=np.array(lat[5:])
out=dict(n=len(rec), runtime_failures=int(sum(x["runtime"] is None for x in rec)),
 logit_absdiff=dict(max=float(d.max()), median=float(np.median(d)), p90=float(np.percentile(d,90))),
 face_rows_absdiff=dict(max=float(okd.max()), median=float(np.median(okd))),
 logit_spread_reference=float(np.std([x["offline"] for x in rec])),
 latency_ms=dict(p50=float(np.percentile(lat,50)), p95=float(np.percentile(lat,95)), max=float(lat.max()), n=int(len(lat))),
 rows=rec)
Path(r"D:\bumbleclaw_v6r1\results\e21_lora_parity_latency.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))

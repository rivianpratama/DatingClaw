import numpy as np, pandas as pd, json
from pathlib import Path
meta=pd.read_csv(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935\samples.csv',low_memory=False)
meta['sample_id']=meta['sample_id'].astype(str)
y=meta['y'].to_numpy(np.int8); old=meta['old_score'].to_numpy(np.float64); sid=meta['sample_id'].to_numpy()
old_imp=np.where(np.isfinite(old),old,50.0)
idx={s:i for i,s in enumerate(sid)}
base_dir=Path(r'D:\bumbleclaw_v6r1\results\e01r_dev4_base\predictions')
arc_dir=Path(r'D:\bumbleclaw_v6r1\results\e01r_dev4_arc_portrait\predictions')
def load(fn):
 d=np.load(fn); ids=d['sample_id'].astype(str); log=d['logit'].astype(float); return ids,log
def scores(fn):
 ids,log=load(fn); s=np.full(len(sid),np.nan); ok=np.array([i in idx for i in ids]); s[np.array([idx[i] for i in ids[ok]])]=log[ok]; return s
# pooled T policy; choose L on H per fold for severe budget
for budget in [2,5,10,20]:
 rows=[]
 for f in ['F2','F3','F4','F5']:
  baseT=scores(base_dir/f'{f}_T_s0.npz'); arcT=scores(arc_dir/f'{f}_T_s0.npz')
  baseH=scores(base_dir/f'{f}_H_s0.npz'); arcH=scores(arc_dir/f'{f}_H_s0.npz')
  # find L on H: select top H by base, top L by arc; H=K-L; severe<=budget
  # H is 4540 rows with K=944. Use all H rows. We need the H rows indices. Load H ids.
  idsH,_=load(base_dir/f'{f}_H_s0.npz'); hrows=np.array([idx[i] for i in idsH])
  # base/arc scores on hrows
  bh=baseH[hrows]; ah=arcH[hrows]; yh=y[hrows]; oh=old_imp[hrows]
  K=944
  hi=np.where(oh>=50)[0]; lo=np.where(oh<50)[0]
  hi_order=hi[np.argsort(-bh[hi])]; lo_order=lo[np.argsort(-ah[lo])]
  bestL=0; bestHits=-1
  for L in range(0,61):
   H=K-L
   sel=np.r_[hi_order[:H],lo_order[:L]]
   sev=int(((oh[sel]<50)&(yh[sel]==0)).sum())
   if sev<=budget:
    hits=int(yh[sel].sum())
    if hits>bestHits: bestHits=hits; bestL=L
  # apply to T
  idsT,_=load(base_dir/f'{f}_T_s0.npz'); trows=np.array([idx[i] for i in idsT])
  bt=baseT[trows]; at=arcT[trows]; yt=y[trows]; ot=old_imp[trows]
  hi=np.where(ot>=50)[0]; lo=np.where(ot<50)[0]
  hi_order=hi[np.argsort(-bt[hi])]; lo_order=lo[np.argsort(-at[lo])]
  H=K-bestL; sel=np.r_[hi_order[:H],lo_order[:bestL]]
  hits=int(yt[sel].sum()); sev=int(((ot[sel]<50)&(yt[sel]==0)).sum())
  rows.append((f,bestL,hits,sev,int(yt.sum())))
 print('budget',budget,rows,'pooled',sum(r[2] for r in rows),'/',sum(r[4] for r in rows),'sev',sum(r[3] for r in rows))

import numpy as np, pandas as pd, joblib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratch.train_50k_grand_ensemble import ConformalGMMGeometricVeto
import __main__
__main__.ConformalGMMGeometricVeto = ConformalGMMGeometricVeto
meta=pd.read_csv(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935\samples.csv',low_memory=False)
meta['sample_id']=meta['sample_id'].astype(str); idx={s:i for i,s in enumerate(meta['sample_id'])}
y=meta['y'].to_numpy(np.int8); old=np.where(np.isfinite(meta['old_score'].to_numpy(float)),meta['old_score'].to_numpy(float),50.0)
def load_pred(fn):
 d=np.load(fn); ids=d['sample_id'].astype(str); log=d['logit'].astype(float); rows=np.array([idx[i] for i in ids]); return rows,log
base='D:/bumbleclaw_v6r1/results/e01r_dev4_base/predictions/'
# hard veto flags from stores
tr=np.load(r'D:\master_50k_dinov3_store.npz',allow_pickle=True); ho=np.load(r'D:\holdout_4540_store.npz',allow_pickle=True)
hard_train=tr['hard_veto'].astype(bool); hard_hold=ho['hard_veto'].astype(bool); hard=np.r_[hard_train,hard_hold]
geom=np.vstack([tr['X_geom'],ho['X_geom']]).astype(np.float32)
veto_model=joblib.load('models/bumbleclaw_conformal_gmm_veto.joblib')
gmm=veto_model.veto(geom).astype(bool)
combined=hard|gmm
print('veto counts',hard.sum(),gmm.sum(),combined.sum())
def soft(log,veto,lam,gam):
 p=1/(1+np.exp(-log)); return log - lam*(1-p)**gam*veto.astype(float)
for name,fn in [('H','F5_H_s0.npz'),('C','F5_C_s0.npz'),('P','F5_P_s0.npz'),('T','F5_T_s0.npz')]:
 rows,log=load_pred(base+fn); K=int(np.floor(.208*len(rows)))
 print(name,'N',len(rows),'K',K,'base hits',int(y[rows][np.argsort(-log)[:K]].sum()),'base sev',int(((old[rows][np.argsort(-log)[:K]]<50)&(y[rows][np.argsort(-log)[:K]]==0)).sum()))
# choose lambda/gamma on H for budget 10 on combined veto
rows,log=load_pred(base+'F5_H_s0.npz'); K=int(np.floor(.208*len(rows))); v=combined[rows]
best=None
for lam in [0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5,8,12]:
 for gam in [1,1.5,2,2.5,3,4]:
  s=soft(log,v,lam,gam); order=np.argsort(-s)[:K]; hits=int(y[rows][order].sum()); sev=int(((old[rows][order]<50)&(y[rows][order]==0)).sum())
  if sev<=10 and (best is None or hits>best[0]): best=(hits,sev,lam,gam)
print('H best combined',best)
if best:
 lam,gam=best[2],best[3]
 for name,fn in [('C','F5_C_s0.npz'),('P','F5_P_s0.npz'),('T','F5_T_s0.npz')]:
  rows,log=load_pred(base+fn); K=int(np.floor(.208*len(rows))); s=soft(log,combined[rows],lam,gam); order=np.argsort(-s)[:K]
  print(name,'hits',int(y[rows][order].sum()),'/',int(y[rows].sum()),'sev',int(((old[rows][order]<50)&(y[rows][order]==0)).sum()),'K',K)

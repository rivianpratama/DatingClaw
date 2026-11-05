import numpy as np, pandas as pd, json
meta=pd.read_csv(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935\samples.csv',low_memory=False)
meta['sample_id']=meta['sample_id'].astype(str); idx={s:i for i,s in enumerate(meta['sample_id'])}
y=meta['y'].to_numpy(np.int8); old=np.where(np.isfinite(meta['old_score'].to_numpy(float)),meta['old_score'].to_numpy(float),50.0)
def load(fn):
 d=np.load(fn); ids=d['sample_id'].astype(str); log=d['logit'].astype(float); return np.array([idx[i] for i in ids]),log
base='D:/bumbleclaw_v6r1/results/e01r_dev4_base/predictions/'
H=load(base+'F5_H_s0.npz'); C=load(base+'F5_C_s0.npz'); P=load(base+'F5_P_s0.npz'); T=load(base+'F5_T_s0.npz')
# soft veto: logit - lambda * (1-p)^gamma * energy ; energy=relu((50-old)/8)
def soft(log,old,lam,gam):
 p=1/(1+np.exp(-log)); e=np.maximum(0,(50-old)/8.0); return log - lam*(1-p)**gam*e
# choose lambda/gamma on H for budget 10 severe (H N=4540 K=944)
best=None
for lam in [0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0]:
 for gam in [1.0,1.5,2.0,2.5,3.0]:
  r,log=H; K=int(np.floor(.208*len(r))); s=soft(log,old[r],lam,gam); order=np.argsort(-s)[:K]
  hits=int(y[r[order]].sum()); sev=int(((old[r[order]]<50)&(y[r[order]]==0)).sum())
  if sev<=10 and (best is None or hits>best[0]): best=(hits,sev,lam,gam)
print('H best',best)
if best:
 lam,gam=best[2],best[3]
 for name,(r,log) in [('C',C),('P',P),('T',T)]:
  K=int(np.floor(.208*len(r))); s=soft(log,old[r],lam,gam); order=np.argsort(-s)[:K]
  hits=int(y[r[order]].sum()); sev=int(((old[r[order]]<50)&(y[r[order]]==0)).sum())
  print(name,'hits',hits,'/',int(y[r].sum()),'sev',sev,'K',K)
 # base raw T
 r,log=T; K=int(np.floor(.208*len(r))); order=np.argsort(-log)[:K]
 print('base raw T','hits',int(y[r[order]].sum()),'sev',int(((old[r[order]]<50)&(y[r[order]]==0)).sum()))

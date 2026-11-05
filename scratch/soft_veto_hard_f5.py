import numpy as np, pandas as pd
meta=pd.read_csv(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935\samples.csv',low_memory=False)
meta['sample_id']=meta['sample_id'].astype(str); idx={s:i for i,s in enumerate(meta['sample_id'])}
y=meta['y'].to_numpy(np.int8); old=np.where(np.isfinite(meta['old_score'].to_numpy(float)),meta['old_score'].to_numpy(float),50.0)
d=np.load('D:/bumbleclaw_v6r1/results/v52_scores_f5.npz'); hv=d['hard_veto'].astype(bool); gv=d['gmm_veto'].astype(bool); cv=d['combined_veto'].astype(bool)
r=np.arange(len(y)); log=np.load('D:/bumbleclaw_v6r1/results/pw10_scores.npz')['base_1'].astype(float)
K=944; order=np.argsort(-log)[:K]
print('base raw',int(y[order].sum()),int(((old[order]<50)&(y[order]==0)).sum()))
for name,v in [('hard',hv),('gmm',gv),('combined',cv)]:
 print(name,'veto count',v.sum(),'likes vetoed',int(y[v].sum()),'severe vetoed',int(((old[v]<50)&(y[v]==0)).sum()))
 # hard removal
 keep=~v; order=np.argsort(-log[keep])[:K]
 print(name,'removed raw',int(y[keep][order].sum()),int(((old[keep][order]<50)&(y[keep][order]==0)).sum()))
 # soft shield
 for lam in [0.5,1,2,4,8]:
  p=1/(1+np.exp(-log)); s=log - lam*(1-p)**2.5*v.astype(float); order=np.argsort(-s)[:K]
  hits=int(y[order].sum()); sev=int(((old[order]<50)&(y[order]==0)).sum())
  print(' soft lam',lam,'hits',hits,'sev',sev)

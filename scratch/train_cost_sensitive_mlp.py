from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bumbleclaw_v6.data.load import LegacyFeatureBank
from bumbleclaw_v6.data.transforms import BlockSpec, FeatureTransform
from bumbleclaw_v6.eval.metrics import deck_metrics
snap=Path(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935')
meta=pd.read_csv(snap/'samples.csv',low_memory=False)
y=meta['y'].to_numpy(np.int8); old=meta['old_score'].to_numpy(np.float64); ids=meta['sample_id'].astype(str).to_numpy()
train_end=45530; val=np.arange(45530,50070,dtype=np.int64); test=np.arange(50070,54610,dtype=np.int64); train=np.arange(0,train_end,dtype=np.int64)
keys=('arc','sf','sp','d3','d2','geom','fv')
specs=[BlockSpec('arc','unit',True,True),BlockSpec('sf','unit',True,True),BlockSpec('sp','unit',True,True),BlockSpec('d3','unit',True,True),BlockSpec('d2','unit',True,True),BlockSpec('geom','standardize',False,True),BlockSpec('fv','passthrough',False,False)]
bank=LegacyFeatureBank(snap,keys); blocks=bank.blocks(keys)
tr=FeatureTransform.fit(blocks,train,specs); X,_=tr.transform(blocks)
old_imp=np.where(np.isfinite(old),old,50.0)
oldn=(old_imp/100.0).astype(np.float32).reshape(-1,1); X=np.hstack([X,oldn]).astype(np.float32)
print('X',X.shape)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(alpha,beta,seed=0,epochs=20,hidden=512,dropout=0.2,wd=1e-4):
 torch.manual_seed(seed); np.random.seed(seed)
 model=nn.Sequential(nn.Linear(X.shape[1],hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1)).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=wd)
 xt=torch.from_numpy(X[train]).to(device); yt=torch.from_numpy(y[train].astype(np.float32)).to(device)
 oldt=torch.from_numpy(old_imp[train].astype(np.float32)).to(device)
 wt=torch.ones(len(train),device=device)
 wt[(yt<0.5)&(oldt<50)] = alpha
 wt[(yt>0.5)&(oldt<50)] = beta
 gen=torch.Generator(device=device).manual_seed(seed); n=len(train)
 for ep in range(epochs):
  order=torch.randperm(n,device=device,generator=gen)
  total=0.0
  for st in range(0,n,1024):
   b=order[st:st+1024]; logits=model(xt[b]).squeeze(1)
   loss=F.binary_cross_entropy_with_logits(logits,yt[b],weight=wt[b])
   opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); total+=float(loss)*len(b)
  if ep%5==4: print(' ep',ep+1,'loss',total/n)
 model.eval()
 with torch.no_grad():
  sv=model(torch.from_numpy(X[val]).to(device)).squeeze(1).cpu().numpy()
  st=model(torch.from_numpy(X[test]).to(device)).squeeze(1).cpu().numpy()
 return sv.astype(np.float32), st.astype(np.float32)
res={}; scores={}
for alpha in [1.0,2.0,4.0,8.0]:
 for beta in [1.0,2.0]:
  key=f'a{alpha}_b{beta}'
  print('training',key)
  sv,st=train_model(alpha,beta)
  mv=deck_metrics(sv,y[val],ids[val],old_score=old[val])
  mt=deck_metrics(st,y[test],ids[test],old_score=old[test])
  # low subset top L on test
  low_test=test[old[test]<50]; low_rel=low_test-test[0]; order=np.argsort(st[low_rel])[::-1]
  lows=[]
  for L in [10,20,30,50,100]:
   top=low_test[order[:L]]; lows.append((L,int(y[top].sum()),int(((old[top]<50)&(y[top]==0)).sum())))
  res[key]={'val':{'hits':mv['hits'],'severe':mv['proxy_severe_fp'],'auc':mv['auc']},'test':{'hits':mt['hits'],'severe':mt['proxy_severe_fp'],'auc':mt['auc']},'low_top':lows}
  scores[key]=st
  print(key,res[key])
np.savez_compressed('D:/bumbleclaw_v6r1/results/cost_sensitive_mlp_scores.npz',**scores)
with open('D:/bumbleclaw_v6r1/results/cost_sensitive_mlp.json','w') as f: json.dump(res,f,indent=2)
print('saved')

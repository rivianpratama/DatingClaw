from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bumbleclaw_v6.data.load import LegacyFeatureBank, load_snapshot
from bumbleclaw_v6.data.transforms import BlockSpec, FeatureTransform
from bumbleclaw_v6.models.linear import fit_linear_logreg
from bumbleclaw_v6.eval.metrics import deck_metrics

snap=Path(r'D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935')
meta=pd.read_csv(snap/'samples.csv',low_memory=False)
y=meta['y'].to_numpy(np.int8); old=meta['old_score'].to_numpy(np.float64); ids=meta['sample_id'].astype(str).to_numpy()
train=np.arange(0,50070,dtype=np.int64); test=np.arange(50070,54610,dtype=np.int64)
low_train=train[old[train]<50]; low_test=test[old[test]<50]
low_test_rel=low_test-test[0]
print('low train',len(low_train),y[low_train].sum(),'low test',len(low_test),y[low_test].sum())
configs={
 'base': (('arc','sf','sp','d3','fv'), [BlockSpec('arc','unit',True,True),BlockSpec('sf','unit',True,True),BlockSpec('sp','unit',True,True),BlockSpec('d3','unit',True,True),BlockSpec('fv','passthrough',False,False)]),
 'all': (('arc','sf','sp','d3','d2','geom','fv'), [BlockSpec('arc','unit',True,True),BlockSpec('sf','unit',True,True),BlockSpec('sp','unit',True,True),BlockSpec('d3','unit',True,True),BlockSpec('d2','unit',True,True),BlockSpec('geom','standardize',False,True),BlockSpec('fv','passthrough',False,False)]),
}
results={}
scores={}
for cname,(keys,specs) in configs.items():
 print('config',cname,keys)
 bank=LegacyFeatureBank(snap,keys); blocks=bank.blocks(keys)
 tr=FeatureTransform.fit(blocks, train, specs)
 X,_=tr.transform(blocks)
 print('X',X.shape)
 for pw in [1.0,5.0,10.0,20.0,40.0,80.0]:
  model=fit_linear_logreg(X[low_train],y[low_train],weight_decay=1.0,seed=0,epochs=20,positive_weight=pw)
  s=model.predict(X[test])
  scores[f'{cname}_pw{pw}']=s.astype(np.float32)
  # metrics all F5
  m=deck_metrics(s,y[test],ids[test],old_score=old[test])
  # low subset top L precision
  order=np.argsort(s[low_test_rel])[::-1]
  lows=[]
  for L in [10,15,20,25,30,40,50,75,100]:
   top=low_test[order[:L]]
   lows.append((L,int(y[top].sum()),int(((old[top]<50)&(y[top]==0)).sum())))
  row={'config':cname,'pw':pw,'hits':m['hits'],'severe':m['proxy_severe_fp'],'auc':m['auc'],'low_top':lows,'model_hash':model.hash}
  results[f'{cname}_pw{pw}']=row
  print(row)
np.savez_compressed('D:/bumbleclaw_v6r1/results/low_aesthetic_probe_scores.npz', **scores)
with open('D:/bumbleclaw_v6r1/results/low_aesthetic_probe.json','w') as f: json.dump(results,f,indent=2)
print('saved')

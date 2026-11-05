"""Fixed-threshold evaluation of the labeled 4k deck. Threshold chosen on first half, reported on second half."""
import csv, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

D=Path(r"D:\BumbleLog\untrained_test_2000")
OUT=Path(r"D:\bumbleclaw_v6r1\results\untrained_4k_eval.json")
sc={}
with (D/"scores.csv").open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f): sc[r["screenshot"]]=r
rows=[]
with (D/"ground_truth_labels.csv").open(encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        s=sc.get(r["filename"])
        if not s: continue
        rows.append(dict(iter=int(s["iteration"]), y=1 if r["action"]=="right" else 0,
                         gambler=float(s["gambler_score"]), v6r5=float(s["v6r5_logit"]),
                         lora=float(s["lora_logit"]), filename=r["filename"]))
rows.sort(key=lambda r:r["iter"]); n=len(rows); half=n//2
dev,test=rows[:half],rows[half:]
def arr(rs,k): return np.array([r[k] for r in rs],float), np.array([r["y"] for r in rs],int)
def wilson(k,n,z=1.96):
    if n==0: return None
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round((c-h)*100,2),round((c+h)*100,2)]
def best_threshold(score,y):
    # max F1 over candidate midpoints; threshold = score >= t
    cand=np.unique(np.quantile(score,np.linspace(0.01,0.99,199)))
    best=(-1,None,None)
    for t in cand:
        pred=score>=t; tp=int(((pred)&(y==1)).sum()); fp=int(((pred)&(y==0)).sum()); fn=int(((~pred)&(y==1)).sum())
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0
        if f1>best[0]: best=(f1,float(t),(prec,rec,tp,fp,fn))
    return best
def evalmodel(key):
    sd,yd=arr(dev,key); st,yt=arr(test,key)
    f1,t,(p,r,tp,fp,fn)=best_threshold(sd,yd)
    pred=st>=t; tp2=int((pred&(yt==1)).sum()); fp2=int((pred&(yt==0)).sum()); fn2=int(((~pred)&(yt==1)).sum()); tn2=int(((~pred)&(yt==0)).sum())
    prec=tp2/(tp2+fp2) if tp2+fp2 else 0; rec=tp2/(tp2+fn2) if tp2+fn2 else 0
    return dict(key=key, threshold=round(t,4), dev_f1=round(f1,4),
                test=dict(n=len(test), likes=int(yt.sum()), tp=tp2, fp=fp2, fn=fn2, tn=tn2,
                          precision=round(prec,4), recall=round(rec,4),
                          precision_ci=wilson(tp2,tp2+fp2), recall_ci=wilson(tp2,tp2+fn2),
                          right_rate=round((tp2+fp2)/len(test),4),
                          auc=round(roc_auc_score(yt,st),4), ap=round(average_precision_score(yt,st),4)))
res={k:evalmodel(k) for k in ("gambler","v6r5","lora")}
out=dict(n=n, likes=int(sum(r["y"] for r in rows)), prevalence=round(sum(r["y"] for r in rows)/n,4),
         dev=dict(n=len(dev),likes=int(sum(r["y"] for r in dev))), test=dict(n=len(test),likes=int(sum(r["y"] for r in test))),
         models=res)
OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))

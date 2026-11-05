"""E23: grouped 5-fold OOF for tight / sp_wide / sf linear models; fixed 0.5/0.3/0.2 z-score blend.
Weights were fixed by the owner before any result. Evaluated on 12 consecutive 4,540-card decks; lambda=1.0 fixed."""
import json, time, numpy as np
from pathlib import Path
from bumbleclaw_v6.data.load import load_snapshot, LegacyFeatureBank
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.experiments.baselines import specs_for
from bumbleclaw_v6.experiments.full_release import _groups, grouped_folds
from bumbleclaw_v6.models.linear import fit_linear_logreg
from bumbleclaw_v6.decision.calibration import fit_platt
from bumbleclaw_v6.decision.policy import select_utility, UnknownCostPolicy
from bumbleclaw_v6.provenance import deterministic_topk
SNAP=Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
OUT=Path(r"D:\bumbleclaw_v6r1\results\e23_blend_oof"); OUT.mkdir(exist_ok=True)
meta=load_snapshot(SNAP)["samples"]; y=meta["y"].to_numpy(np.int8); ids=meta["sample_id"].astype(str).to_numpy()
old=meta["old_score"].to_numpy(float); n=len(y)
folds=grouped_folds(_groups(meta),5,0)
crop=FeatureStoreBank(SNAP,Path(r"D:\bumbleclaw_v6r1\features\crop_v1_full"),get_recipe("crop_v1"))
models={"tight":(crop,("arc","sp_tight","d3","fv")),"wide":(crop,("sp_wide","fv")),"sf":(LegacyFeatureBank(SNAP,("sf",)),("sf",))}
oof={}
for name,(bank,keys) in models.items():
    p=OUT/f"oof_{name}.npy"
    if p.exists(): oof[name]=np.load(p); continue
    t=time.time(); blocks=bank.blocks(keys); specs=specs_for(keys); o=np.full(n,np.nan)
    for f in range(5):
        tr=np.flatnonzero(folds!=f); va=np.flatnonzero(folds==f)
        tf=FeatureTransform.fit(blocks,tr,specs); X,_=tf.transform(blocks)
        o[va]=fit_linear_logreg(X[tr],y[tr],weight_decay=1.0,seed=0).predict(X[va])
    assert np.isfinite(o).all(); np.save(p,o); oof[name]=o; print(name,"done",round(time.time()-t),"s",flush=True)
z={k:(v-v.mean())/v.std() for k,v in oof.items()}
cands={"tight":z["tight"],"wide":z["wide"],"sf":z["sf"],
       "blend_tight_50_30_20":0.5*z["tight"]+0.3*z["wide"]+0.2*z["sf"],
       "blend_tight_wide_only_62_38":0.625*z["tight"]+0.375*z["wide"]}
decks=[np.arange(i*4540,(i+1)*4540) for i in range(n//4540)]
def evaluate(s):
    cal=fit_platt(s,y,model_hash="e23",transform_hash="e23",recipe_hash="e23",calibration_ids=ids)
    pr=cal.apply(s,model_hash="e23",transform_hash="e23",recipe_hash="e23")
    per=[]
    for d in decks:
        k=int(np.floor(0.208*len(d)))
        raw=d[np.asarray(deterministic_topk(s[d],ids[d],k))]
        pol=d[np.asarray(select_utility(pr[d],old[d],ids[d],k,1.0,unknown_policy=UnknownCostPolicy.ABSTAIN).selected)]
        f=lambda sel:(int(y[sel].sum()),int(((y[sel]==0)&(old[sel]<50)).sum()))
        per.append(f(raw)+f(pol))
    return np.array(per)
res={k:evaluate(v) for k,v in cands.items()}
rng=np.random.default_rng(0); B=rng.integers(0,len(decks),(5000,len(decks)))
def ci(a,b,col):
    d=(res[a][:,col]-res[b][:,col]); bs=d[B].sum(1); return [int(d.sum()),int(np.percentile(bs,2.5)),int(np.percentile(bs,97.5))]
summary={"protocol":"grouped 5-fold OOF linear (wd=1.0), z-blend fixed weights, 12 decks x 4540, K=944/deck, lambda=1.0, Platt on OOF",
 "n_pos_total":int(sum(y[d].sum() for d in decks)),
 "totals":{k:{"raw_hits":int(v[:,0].sum()),"raw_severe":int(v[:,1].sum()),"policy_hits":int(v[:,2].sum()),"policy_severe":int(v[:,3].sum())} for k,v in res.items()},
 "paired_vs_tight_deck_bootstrap_95ci":{k:{"policy_hits":ci(k,"tight",2),"policy_severe":ci(k,"tight",3)} for k in cands if k!="tight"}}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

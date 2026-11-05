"""E24: fold the fixed 0.625 tight + 0.375 wide z-blend into one linear release (photo-only)."""
import json, shutil, numpy as np
from pathlib import Path
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.experiments.baselines import specs_for
from bumbleclaw_v6.models.linear import LinearModel, fit_linear_logreg
from bumbleclaw_v6.decision.calibration import fit_platt
from bumbleclaw_v6.decision.policy import select_utility, UnknownCostPolicy
from bumbleclaw_v6.provenance import stable_hash, sha256_file
from bumbleclaw_v6.release.validate import validate_release
SNAP=Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
V62=Path(r"D:\bumbleclaw_v6r1\releases\v6r2_photo_tight_full"); E23=Path(r"D:\bumbleclaw_v6r1\results\e23_blend_oof")
OUT=Path(r"D:\bumbleclaw_v6r1\releases\v6r5_photo_tight_wide_blend"); assert not OUT.exists()
WT,WW=0.625,0.375
meta=load_snapshot(SNAP)["samples"]; y=meta["y"].to_numpy(np.int8); ids=meta["sample_id"].astype(str).to_numpy(); old=meta["old_score"].to_numpy(float)
n=len(y); allr=np.arange(n)
bank=FeatureStoreBank(SNAP,Path(r"D:\bumbleclaw_v6r1\features\crop_v1_full"),get_recipe("crop_v1"))
KT=("arc","sp_tight","d3","fv"); KW=("sp_wide","fv"); KC=("arc","sp_tight","d3","fv","sp_wide")
blocks=bank.blocks(KC)
# tight part = frozen v6.2 model + transform (exact)
mt=LinearModel.load(V62/"model.npz"); tt=FeatureTransform.from_dict(json.loads((V62/"transform.json").read_text()))
Xt,nt=tt.transform({k:blocks[k] for k in KT})
tw=FeatureTransform.fit({k:blocks[k] for k in KW},allr,specs_for(KW)); Xw,nw=tw.transform({k:blocks[k] for k in KW})
mw=fit_linear_logreg(Xw,y,weight_decay=1.0,seed=0)
tc=FeatureTransform.fit(blocks,allr,specs_for(KC)); Xc,nc=tc.transform(blocks)
assert np.array_equal(Xc[:,[nc.index(c) for c in nt]],Xt) and np.array_equal(Xc[:,[nc.index(c) for c in nw]],Xw)
ot,ow=np.load(E23/"oof_tight.npy"),np.load(E23/"oof_wide.npy")
mT,sT,mW,sW=ot.mean(),ot.std(),ow.mean(),ow.std()
w=np.zeros(Xc.shape[1],np.float64)
for c,v in zip(nt,mt.weights): w[nc.index(c)]+=WT*v/sT
for c,v in zip(nw,mw.weights): w[nc.index(c)]+=WW*v/sW
bias=WT*(mt.bias-mT)/sT+WW*(mw.bias-mW)/sW
model=LinearModel(weights=w.astype(np.float32),bias=float(bias),metadata={"kind":"fixed_z_blend","weights":{"tight":WT,"wide":WW},
  "z_stats_source":"grouped 5-fold OOF logits (e23)","z":{"tight":[mT,sT],"wide":[mW,sW]},"tight_part":"v6r2_photo_tight_full model.npz","wide_part":"fit all rows wd=1.0 seed0"})
lc=model.predict(Xc); ref=WT*(mt.predict(Xt)-mT)/sT+WW*(mw.predict(Xw)-mW)/sW
fold_err=float(np.abs(lc-ref).max()); assert fold_err<1e-3, fold_err
oof_blend=WT*(ot-mT)/sT+WW*(ow-mW)/sW
recipe=get_recipe("photo_tight_wide_v1"); recipe_hash=stable_hash({"recipe":recipe.name,"recipe_hash":recipe.hash,"keys":list(KC)})
th=stable_hash(tc.to_dict())
cal=fit_platt(oof_blend,y,model_hash=model.hash,transform_hash=th,recipe_hash=recipe_hash,calibration_ids=ids)
# reproduce E23 policy totals with the bound calibrator (sanity)
p=cal.apply(oof_blend,model_hash=model.hash,transform_hash=th,recipe_hash=recipe_hash); H=S=0
for i in range(n//4540):
    d=np.arange(i*4540,(i+1)*4540); sel=d[np.asarray(select_utility(p[d],old[d],ids[d],944,1.0,unknown_policy=UnknownCostPolicy.ABSTAIN).selected)]
    H+=int(y[sel].sum()); S+=int(((y[sel]==0)&(old[sel]<50)).sum())
OUT.mkdir(parents=True)
model.save(OUT/"model.npz"); (OUT/"transform.json").write_text(json.dumps(tc.to_dict()))
dec=json.loads((V62/"decision.json").read_text()); dec["calibrator"]=cal.to_dict()
known=np.flatnonzero(np.isfinite(old)); recent=known[-300:]
u=p[recent]-1.0*(1-p[recent])*(old[recent]<50).astype(float)
dec["controller"]["warm_start"]=[float(v) for v in u]
dec["controller"]["warm_start_source"]="OOF calibrated utilities of the 300 most recent known-cost cards"
(OUT/"decision.json").write_text(json.dumps(dec)); (OUT/"calibration.json").write_text(json.dumps(cal.to_dict()))
pol=json.loads((V62/"policy.json").read_text()); pol["note"]="fixed lambda=1.0 copied from v6r2; blend weights owner-fixed before results"
(OUT/"policy.json").write_text(json.dumps(pol))
files={f:sha256_file(OUT/f) for f in ("model.npz","transform.json","calibration.json","decision.json","policy.json")}
man={"name":OUT.name,"kind":"photo_only_full_data_release","keys":list(KC),"recipe":recipe.name,"recipe_hash":recipe_hash,
 "model_hash":model.hash,"transform_hash":th,"files":files,"photo_only":True,"uses_sf":False,"feature_store":str(bank.store.root) if hasattr(bank.store,"root") else r"D:\bumbleclaw_v6r1\features\crop_v1_full",
 "blend":{"tight":WT,"wide":WW},"fold_error_max":fold_err,"oof_policy_check_12_decks":{"hits":H,"severe":S},
 "evidence":"E23 grouped OOF development comparison; not a sealed test","calibration_method":"grouped_out_of_fold_platt"}
(OUT/"manifest.json").write_text(json.dumps(man,indent=2))
r=validate_release(OUT); print(json.dumps({"PASS":r["PASS"],"fold_err":fold_err,"oof_policy":[H,S],"cal":[cal.a,cal.b]}))

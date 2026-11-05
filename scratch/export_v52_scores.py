from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, torch, joblib
os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"; os.environ["TMP"]=r"D:\BumbleTemp"; os.environ["TMPDIR"]=r"D:\BumbleTemp"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratch.train_50k_grand_ensemble import (
    ConformalGMMGeometricVeto, MultiXSafeMetaBlender,
    Pillar1SingleCropNet as P1_v4, Pillar2DualCropNet as P2_v4,
    Pillar3SubcenterNet as P3_v4, Pillar4OrthogonalRankNet as P4_v4,
)
import __main__
__main__.MultiXSafeMetaBlender = MultiXSafeMetaBlender
__main__.ConformalGMMGeometricVeto = ConformalGMMGeometricVeto
from scratch.extract_and_train_dinov3 import (
    Pillar1SingleCropNet as P1_d3, Pillar2DualCropNet as P2_d3,
    Pillar3SubcenterNet as P3_d3, Pillar4OrthogonalRankNet as P4_d3,
)
root=Path(__file__).resolve().parent.parent; models=root/'models'; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hold=np.load(r'D:\holdout_4540_store.npz',allow_pickle=True)
X_arc=hold['X_arc']; X_geom=hold['X_geom']; X_sf=hold['X_sig_full']; X_sp=hold['X_sig_port']; X_d2=hold['X_dino2']; X_d3=hold['X_dino3']; fv=hold['face_valid']; hard=hold['hard_veto']; y=hold['y']; mx=hold['s_multi_x']
t_arc=torch.tensor(X_arc,device=device); t_geom=torch.tensor(X_geom,device=device); t_sf=torch.tensor(X_sf,device=device); t_sp=torch.tensor(X_sp,device=device); t_d2=torch.tensor(X_d2,device=device); t_d3=torch.tensor(X_d3,device=device); t_fv=torch.tensor(fv,device=device)
veto_model=joblib.load(models/'bumbleclaw_conformal_gmm_veto.joblib'); gmm=veto_model.veto(X_geom); combined=hard|gmm
out={}
with torch.no_grad():
 p1=P1_v4().to(device); p1.load_state_dict(torch.load(models/'bumbleclaw_v4_p1_50k.pt',map_location=device)); p1.eval()
 p2=P2_v4().to(device); p2.load_state_dict(torch.load(models/'bumbleclaw_v4_p2_50k.pt',map_location=device)); p2.eval()
 p3=P3_v4(K=16,M=48).to(device); p3.load_state_dict(torch.load(models/'bumbleclaw_v4_p3_50k.pt',map_location=device)); p3.eval()
 p4=P4_v4().to(device); p4.load_state_dict(torch.load(models/'bumbleclaw_v4_p4_50k.pt',map_location=device)); p4.eval()
 meta=joblib.load(models/'bumbleclaw_v4_meta_blender_50k.joblib')
 z1=p1(t_arc,t_geom,t_fv).cpu().numpy(); z2=p2(t_arc,t_geom,t_sf,t_sp,t_d2,t_fv).cpu().numpy(); z3=p3.compute_logits(p3.embed(t_arc,t_geom,t_fv)).cpu().numpy(); z4=p4(t_arc,t_geom,t_sf,t_sp,t_d2,t_fv).cpu().numpy()
 Z=np.stack([z1,z2,z3,z4],1); s_v4=meta.predict_score(Z,mx); s_v4_veto=np.where(combined,-1e9,s_v4)
 out['v4_raw']=s_v4.astype(np.float32); out['v4_vetoed']=s_v4_veto.astype(np.float32)
 p1=P1_d3().to(device); p1.load_state_dict(torch.load(models/'bumbleclaw_v5_p1_dinov3.pt',map_location=device)); p1.eval()
 p2=P2_d3().to(device); p2.load_state_dict(torch.load(models/'bumbleclaw_v5_p2_dinov3.pt',map_location=device)); p2.eval()
 p3=P3_d3(K=16,M=48).to(device); p3.load_state_dict(torch.load(models/'bumbleclaw_v5_p3_dinov3.pt',map_location=device)); p3.eval()
 p4=P4_d3().to(device); p4.load_state_dict(torch.load(models/'bumbleclaw_v5_p4_dinov3.pt',map_location=device)); p4.eval()
 meta=joblib.load(models/'bumbleclaw_v5_meta_blender_dinov3.joblib')
 z1=p1(t_arc,t_geom,t_fv).cpu().numpy(); z2=p2(t_arc,t_geom,t_sf,t_sp,t_d3,t_fv).cpu().numpy(); z3=p3.compute_logits(p3.embed(t_arc,t_geom,t_fv)).cpu().numpy(); z4=p4(t_arc,t_geom,t_sf,t_sp,t_d3,t_fv).cpu().numpy()
 Z=np.stack([z1,z2,z3,z4],1); raw=meta.predict_score(Z,mx)
 L=meta._calibrated_log_odds(Z); w_p=meta.meta_params[:4]; w_lse=meta.meta_params[4]; bias=meta.meta_params[7]; tau=meta.tau_lse
 mmax=np.max(L/tau,axis=1,keepdims=True); llse=tau*(mmax.squeeze(1)+np.log(np.sum(np.exp(L/tau-mmax),axis=1)))
 s_match=(L@w_p)+w_lse*llse+bias
 is_vetoed=combined|(mx<42.5)|((mx<49.0)&(s_match<0.0))
 out['v52_raw']=raw.astype(np.float32); out['v52_vetoed']=np.where(is_vetoed,-1e9,raw).astype(np.float32)
 out['v52_match']=s_match.astype(np.float32); out['gmm_veto']=gmm.astype(np.uint8); out['hard_veto']=hard.astype(np.uint8); out['combined_veto']=combined.astype(np.uint8)
 out['y']=y.astype(np.uint8); out['multi_x']=mx.astype(np.float32)
np.savez_compressed('D:/bumbleclaw_v6r1/results/v52_scores_f5.npz', **out)
print('saved', {k:v.shape for k,v in out.items()})

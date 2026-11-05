"""Fail unless runtime boxes are complete and face_valid matches the canonical photo_v1 store on every row."""
import sys, numpy as np
from pathlib import Path
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
SNAP=Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
with np.load(r"D:\bumbleclaw_v6r1\features\boxes_runtime_v1.npz", allow_pickle=False) as z: z=dict(z)
ids=np.asarray(load_snapshot(SNAP)["samples"]["sample_id"].astype(str).tolist())
fv=FeatureStoreBank(SNAP,Path(r"D:\bumbleclaw_v6r1\features\photo_v1_full_r2"),get_recipe("photo_v1")).blocks(("fv",))["fv"].values[:,0]>0.5
ok_ids=np.array_equal(z["sample_id"].astype(str), ids); done=bool(z["done"].all())
mism=int((z["face_valid"].astype(bool)!=fv).sum()); badbox=int((np.all(z["bbox"]==-1,axis=1)==z["face_valid"].astype(bool)).sum())  # -1 sentinel = no face; negative x1 is a legit off-edge face
print(f"verify boxes: ids_match={ok_ids} done={done} face_valid_mismatch={mism} bbox_validity_mismatch={badbox} faces={int(z['face_valid'].sum())}/{len(ids)}")
sys.exit(0 if (ok_ids and done and mism==0 and badbox==0) else 1)

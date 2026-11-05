"""Model-based duplicate scan: ArcFace embeddings for profile_*.webp; cosine >= threshold."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from bumbleclaw_v6.features.extract import extract_image, load_models
from bumbleclaw_v6.features.recipes import FeatureRecipe
from bumbleclaw_v6.features.preprocess import PHOTO_V1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.995)
    ap.add_argument("--limit", type=int, default=0)
    args=ap.parse_args()
    files=sorted(args.folder.glob("profile_*.webp"))
    if args.limit: files=files[:args.limit]
    recipe=FeatureRecipe("dup_arc_v1", ("arc","fv"), (PHOTO_V1,))
    models=load_models(recipe, need_arc=True, need_siglip=False, need_dinov3=False)
    E=np.zeros((len(files),512),np.float32); valid=np.zeros(len(files),bool)
    t0=time.time()
    for i,p in enumerate(files):
        b=extract_image(p,recipe,models)
        if b.masks.get("arc",False):
            v=np.asarray(b.values["arc"],np.float32); n=np.linalg.norm(v)
            if n>1e-6: E[i]=v/n; valid[i]=True
        if (i+1)%500==0: print(f"{i+1}/{len(files)} {time.time()-t0:.0f}s",flush=True)
    np.savez(args.out.with_suffix(".emb.npz"), E=E, valid=valid, files=np.array([p.name for p in files]))
    S=E@E.T; np.fill_diagonal(S,-1.0)
    pairs=[]
    for i in range(len(files)):
        if not valid[i]: continue
        js=np.flatnonzero((S[i]>=args.threshold)&valid)
        for j in js:
            if j>i: pairs.append((float(S[i,j]),files[i].name,files[j].name))
    pairs.sort(reverse=True)
    # union-find groups
    parent=list(range(len(files)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[b]=a
    idx={p.name:i for i,p in enumerate(files)}
    for s,a,b in pairs: union(idx[a],idx[b])
    groups={}
    for i in range(len(files)):
        if valid[i]: groups.setdefault(find(i),[]).append(files[i].name)
    groups=[sorted(v) for v in groups.values() if len(v)>1]
    out={"folder":str(args.folder),"n_images":len(files),"n_face":int(valid.sum()),
         "threshold":args.threshold,"n_pairs":len(pairs),"n_groups":len(groups),
         "n_images_in_groups":sum(len(g) for g in groups),
         "groups":groups,"top_pairs":pairs[:50]}
    args.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in ("groups","top_pairs")},indent=2))
    for g in sorted(groups,key=len,reverse=True)[:10]: print(len(g),g[:6])
    print("top pairs",pairs[:5])

if __name__=="__main__": main()

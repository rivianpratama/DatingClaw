"""Detect repeated profile cards by hashing the bottom name/bio region (photo-independent)."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fftpack import dct

def phash_img(im):
    g=im.convert('L').resize((32,32),Image.Resampling.LANCZOS)
    a=np.asarray(g,dtype=np.float32)
    c=dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8]
    med=np.median(c.flatten()[1:]); h=0
    for b in (c.flatten()>med): h=(h<<1)|int(b)
    return h

def rec(path):
    with Image.open(path) as im:
        w,h=im.size
        # bottom text block: names/bio/tags, excluding nav bar
        crop=im.crop((int(.02*w),int(.60*h),int(.98*w),int(.94*h)))
        return path.name, phash_img(crop), w, h

def pc(x): return int(x).bit_count()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('folder',type=Path); ap.add_argument('--csv',type=Path); ap.add_argument('--near',type=int,default=4); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    files=sorted(a.folder.glob('profile_*.webp'))
    with ThreadPoolExecutor(max_workers=8) as ex: recs=list(ex.map(rec,files))
    parent=list(range(len(recs)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        x,y=find(i),find(j)
        if x!=y: parent[y]=x
    for i in range(len(recs)):
        for j in range(i+1,len(recs)):
            if pc(recs[i][1]^recs[j][1])<=a.near: union(i,j)
    groups=defaultdict(list)
    for i in range(len(recs)): groups[find(i)].append(i)
    near=[g for g in groups.values() if len(g)>1]
    meta={}
    if a.csv and a.csv.exists():
        with a.csv.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f): meta[r.get('screenshot','')]={k:r.get(k,'') for k in ('action','gambler_score','iteration')}
    def desc(g):
        names=[recs[i][0] for i in g]
        return {'count':len(names),'files':names,'actions':[meta.get(n,{}).get('action','') for n in names],'scores':[meta.get(n,{}).get('gambler_score','') for n in names]}
    out={'folder':str(a.folder),'n_images':len(recs),'threshold':a.near,'n_groups':len(near),'n_images_in_groups':sum(len(g) for g in near),'groups':[desc(g) for g in near]}
    a.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='groups'},indent=2))
    for g in sorted(out['groups'],key=lambda x:-x['count'])[:15]: print(g['count'],g['files'][:5],g['actions'][:5],g['scores'][:5])
if __name__=='__main__': main()

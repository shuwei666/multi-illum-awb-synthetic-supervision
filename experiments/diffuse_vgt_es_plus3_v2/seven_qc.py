"""Fixed24 permitted Nikon previews; construction only, no evaluation GT."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from seven_synthesis import fields,ARMS

ROOT=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('DIFFUSE_PARENT',str(ROOT.parent/'diffuse_vgt_v1')))
DATA=Path(os.environ.get('NIKON_DATA',str(ROOT.parent/'data/nikon_512')))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('producer',PARENT/'diffuse_synthesis.py')
    producer=importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    manifest=json.loads((PARENT/'source_manifest.json').read_text())
    byid={r['scene_id']:r for r in manifest['sources']}
    out=ROOT/'real24_qc_diagnostics_v2';out.mkdir(exist_ok=False)
    rows=[]
    torch.set_num_threads(4)
    for scene in manifest['qc_selection']['scenes']:
        r=byid[scene]; path=DATA/r['file']; assert sha(path)==r['sha256']
        cache=PARENT/'cache_all'/f'{scene}.npz'
        meta=json.loads((PARENT/'cache_metadata'/f'{scene}.json').read_text())
        assert sha(cache)==meta['cache_sha256']
        raw=cv2.cvtColor(cv2.imread(str(path),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
        w=torch.tensor(raw.astype(np.float32)/np.array(r['Light1'],np.float32)/4).permute(2,0,1)[None]
        w=F.avg_pool2d(w,2)
        with np.load(cache) as z:
            s=F.avg_pool2d(torch.tensor(z['s0'])[None,None],2)
            n=F.avg_pool2d(torch.tensor(z['normal']).permute(2,0,1)[None],2)
        n=n/n.norm(dim=1,keepdim=True).clamp_min(1e-12)
        p=dict(s0=s,normal=n)
        preview=meta['preview']
        L=torch.tensor(preview['lights'])[None];d=torch.tensor(preview['directions'])[None];a=torch.tensor(preview['powers'])[None]
        with torch.no_grad():
            result=fields(producer.render_fields,w,p,L,d,a)
            original=producer.render_fields(w,p,L,d,a)
        E=original['arms']['A']['E'];S=original['arms']['A']['T']
        q=lambda x:torch.flip(x,(-2,-1))
        unit=E/E.norm(dim=1,keepdim=True);qunit=q(unit)
        def rms(x):return float(x.square().mean().sqrt())
        pm=lambda x:F.avg_pool2d(x,16)
        ep=pm(E);ep=ep/ep.norm(dim=1,keepdim=True)
        eqp=pm(q(E));eqp=eqp/eqp.norm(dim=1,keepdim=True)
        row=dict(scene_id=scene,input_sha256=sha(path),cache_sha256=sha(cache),
            E_rotation_mean_deg=float(torch.rad2deg(torch.acos((unit*qunit).sum(1).clamp(-1,1))).mean()),
            S_rotation_log_rms=rms(S.log()-q(S).log()),s0_rotation_log_rms=rms(s.log()-q(s).log()),
            E_rotation_patch_mean_deg=float(torch.rad2deg(torch.acos((ep*eqp).sum(1).clamp(-1,1))).mean()),
            S_rotation_patch_log_rms=rms(pm(S.log())-pm(q(S).log())),
            s0_rotation_patch_log_rms=rms(pm(s.log())-pm(q(s).log())),
            exposure=float(result['exposure'][0]))
        images=[('W',w)]+[(arm,result['arms'][arm]['I']) for arm in ARMS]
        maximum=max(float(x.max()) for _,x in images)
        canvas=Image.new('RGB',(4*256,2*280),'white');draw=ImageDraw.Draw(canvas)
        for i,(name,x) in enumerate(images):
            rgb=x[0].permute(1,2,0).numpy()
            display=np.uint8(np.round(np.clip(rgb/maximum,0,1)**(1/2.2)*255))
            left,top=(i%4)*256,(i//4)*280
            canvas.paste(Image.fromarray(display),(left,top+24));draw.text((left+5,top+5),name,fill='black')
        canvas.save(out/f'{scene}.png');rows.append(row)
    (out/'summary.json').write_text(json.dumps(dict(status='PASS',count=len(rows),scope='fixed24fullview_formula_preview_not_training_crop',scenes=rows),indent=2)+'\n')
    print(json.dumps(dict(count=len(rows),status='PASS')))

if __name__=='__main__':main()

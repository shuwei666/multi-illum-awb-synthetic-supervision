"""Offline fixed-prior cache from allowed Nikon single sources only. QC first."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace, ModuleType
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
for name in ['Intrinsic','chrislib','MiDaS','DSINE']:
    sys.path.insert(0,str(ROOT/'vendor'/name))
from render_diffuse import render_diffuse

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def save(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def load_priors():
    import geffnet
    import torchvision.models as tv
    from intrinsic.pipeline import run_gray_pipeline
    from altered_midas.midas_net import MidasNet
    from altered_midas.midas_net_custom import MidasNet_small
    original=torch.hub.load
    def local_backbone(repo,model,*args,**kw):
        if repo=='facebookresearch/WSL-Images' and model=='resnext101_32x8d_wsl':
            return tv.resnext101_32x8d(weights=None)
        if repo=='rwightman/gen-efficientnet-pytorch' and model=='tf_efficientnet_lite3':
            return geffnet.create_model(model,pretrained=False,exportable=kw.get('exportable',True))
        raise RuntimeError(f'Unapproved backbone request: {repo}/{model}')
    torch.hub.load=local_backbone
    try:
        gray={'ord_model':MidasNet(), 'iid_model':MidasNet_small(exportable=False,input_channels=5,output_channels=1)}
    finally:
        torch.hub.load=original
    weights=torch.load(ROOT/'weights/intrinsic_final_weights.pt',map_location='cpu')
    for key,source in [('ord_model','ord_state_dict'),('iid_model','iid_state_dict')]:
        gray[key].load_state_dict(weights[source],strict=True)
        gray[key].eval().cuda()
    del weights
    # DSINE's namespace utils/ otherwise loses to MiDaS's top-level utils.py.
    # Bind only this inference process; neither vendor tree is edited.
    dsine_utils=ModuleType('utils')
    dsine_utils.__path__=[str(ROOT/'vendor/DSINE/utils')]
    sys.modules['utils']=dsine_utils
    from models.dsine.v02 import DSINE_v02
    args=SimpleNamespace(NNET_encoder_B=5,NNET_decoder_NF=2048,NNET_decoder_BN=False,
         NNET_decoder_down=8,NNET_learned_upsampling=True,NRN_prop_ps=5,NRN_num_iter_train=5,
         NRN_num_iter_test=5,NRN_ray_relu=True,NNET_output_dim=3,NNET_feature_dim=64,NNET_hidden_dim=64)
    original_create=geffnet.create_model
    def untrained(*a,**kw):
        kw['pretrained']=False
        return original_create(*a,**kw)
    geffnet.create_model=untrained
    try: normal=DSINE_v02(args)
    finally: geffnet.create_model=original_create
    state=torch.load(ROOT/'weights/dsine_exp001.pt',map_location='cpu')['model']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    normal.load_state_dict(state,strict=True)
    normal.eval().cuda()
    return gray,normal,run_gray_pipeline

def normal_inference(model,linear):
    from utils.projection import intrins_from_fov
    # Standard sRGB transfer function only as a display encoding of camera RGB;
    # this is NOT a camera-to-sRGB color transform.
    display=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055).astype(np.float32)
    x=torch.from_numpy(display.transpose(2,0,1).copy())[None].cuda()
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    k=intrins_from_fov(60.,linear.shape[0],linear.shape[1],device='cuda')[None]
    with torch.inference_mode():
        n=model((x-mean)/std,intrins=k.clone(),mode='test')[-1][0,:3].permute(1,2,0).cpu().numpy()
    norm=np.linalg.norm(n,axis=2)
    valid=np.isfinite(n).all(2)&np.isfinite(norm)&(norm>1e-8)
    n=np.where(valid[:,:,None],n/np.maximum(norm[:,:,None],1e-8),np.array([0,0,1],np.float32))
    return n.astype(np.float32),valid

def preview(scene,w,s,n,v,pool):
    # No training crop in QC; final synthesis is on the 256² grid.
    w=cv2.resize(w,(256,256),interpolation=cv2.INTER_AREA)
    s=cv2.resize(s,(256,256),interpolation=cv2.INTER_LINEAR)
    n=cv2.resize(n,(256,256),interpolation=cv2.INTER_LINEAR)
    norms=np.linalg.norm(n,axis=2)
    v=(cv2.resize(v.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA)>=1)&(norms>1e-8)
    n=n/np.maximum(norms[:,:,None],1e-8)
    rng=np.random.default_rng(int(hashlib.sha256(('qc:'+scene).encode()).hexdigest()[:16],16))
    lights=pool[rng.integers(0,len(pool),2)]
    phi=rng.uniform(0,2*np.pi,2); z=rng.uniform(np.cos(np.deg2rad(70)),1,2)
    d=np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],axis=1).astype(np.float32)
    r=rng.uniform(-1,1); powers=np.array([2**(r/2),2**(-r/2)],np.float32)
    out=render_diffuse(w.astype(np.float32),s,n,v,lights,d,powers,facing_normal=np.array([0,0,1],np.float32))
    f=float(out['exposure'])
    scale=max(float((w*f).max()),*(float(a['I'].max()) for a in out['arms'].values()),float((w/s[:,:,None]*f).max()))
    def disp(x): return np.uint8(np.clip(np.maximum(x,0)/scale,0,1)**(1/2.2)*255)
    tiles=[('f W',disp(w*f)),('s0 .25..4',np.uint8(np.clip((np.log2(s)+2)/4,0,1)*255)),
           ('normal +Z front',np.uint8(np.clip((n+1)/2,0,1)*255)),('f W/s0',disp(w/s[:,:,None]*f)),
           ('common validity',out['technical_valid'].astype(np.uint8)*255)]
    for name,a in out['arms'].items():
        tiles.extend([(name,disp(a['I'])),(name+' E',np.uint8(np.clip(a['E']/5,0,1)*255))])
    tiles.extend([('H /10',np.uint8(np.clip(out['arms']['A']['H']/10,0,1)*255)),
                  ('T /4',np.uint8(np.clip(out['arms']['A']['T']/4,0,1)*255)),
                  ('alpha',np.uint8(np.clip(out['alpha'],0,1)*255)),
                  ('dark proxy risk',np.uint8(np.max(w,axis=2)<.01*np.max(w))*255),
                  ('J A',disp(out['arms']['A']['J'])),('J B',disp(out['arms']['B']['J'])),
                  ('J C',disp(out['arms']['C']['J']))])
    canvas=Image.new('RGB',(256*6,280*3),'white'); draw=ImageDraw.Draw(canvas)
    for i,(title,img) in enumerate(tiles):
        x=(i%6)*256; y=(i//6)*280
        canvas.paste(Image.fromarray(img).convert('RGB'),(x,y+24)); draw.text((x+4,y+4),scene+' '+title,fill='black')
    canvas.save(ROOT/'qc'/f'{scene}.png')
    technical=out['technical_valid'].reshape(16,16,16,16).all(axis=(1,3))
    parent=(w.max(axis=2).reshape(16,16,16,16).max(axis=(1,3))>0)
    return dict(lights=lights.tolist(),directions=d.tolist(),powers=powers.tolist(),
       common_exposure=float(out['exposure']),display_scale=scale,
       common_pixel_valid_fraction=float(out['technical_valid'].mean()),
       nonblack_qc_patches=int(parent.sum()),common_valid_qc_patches=int((technical&parent).sum()),
       intervention_E_rms=float(np.sqrt(np.mean((out['arms']['B']['E']-out['arms']['C']['E'])**2))))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',required=True); parser.add_argument('--limit',type=int,default=24)
    args=parser.parse_args()
    if not 1<=args.limit<=24: raise ValueError('Only fixed24 QC authorized by this entry point')
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    byid={r['scene_id']:r for r in manifest['sources']}
    scenes=manifest['qc_selection']['scenes'][:args.limit]
    pool=np.array([r['rgb_g1'] for r in manifest['allowed_endpoints']],np.float32)
    (ROOT/'cache_qc').mkdir(exist_ok=True); (ROOT/'qc').mkdir(exist_ok=True)
    priors=json.loads((ROOT/'priors_manifest.json').read_text())
    for name,row in priors['weights'].items():
        assert sha(ROOT/'weights'/name)==row['sha256'],f'Weight hash mismatch: {name}'
    for name,row in priors['repos'].items():
        for path,digest in row['python_files'].items():
            file=ROOT/'vendor'/name/path
            if file.exists(): assert sha(file)==digest,f'Vendor source mismatch: {name}/{path}'
    assert sha(ROOT/'source_manifest.json')==priors['source_manifest_sha256']
    import importlib.metadata as md
    versions={name:md.version(name) for name in ['torch','torchvision','numpy','geffnet','timm','scipy','scikit-image','kornia']}
    gray,normal,run_gray=load_priors()
    save(ROOT/'prior_runtime.json',dict(versions=versions,strict_load_all_models=True,
         priors_manifest_sha256=sha(ROOT/'priors_manifest.json'),cache_code_sha256=sha(Path(__file__)),
         adapter_equivalence='Key/shape strict coverage only; forward equivalence not independently proven'))
    from chrislib.general import uninvert
    checks=np.array([.25,1,4,20.],np.float32)
    np.testing.assert_allclose(uninvert(1/(1+checks)),checks,rtol=1e-5)
    summary=[]; start=time.time()
    for scene in scenes:
        row=byid[scene]; p=Path(args.data)/row['file']
        assert p.name==scene+'_1.tiff' and sha(p)==row['sha256']
        rgb=cv2.cvtColor(cv2.imread(str(p),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype(np.float32)
        w=rgb/np.asarray(row['Light1'],np.float32)
        assert np.isfinite(w).all() and np.max(w/4)<=65504
        scale=float(w.max()); assert scale>0
        proxy=(w/scale).astype(np.float32)
        t=time.time(); output=run_gray(gray,proxy,linear=True,resize_conf=1024,maintain_size=False,device='cuda')
        raw=np.asarray(uninvert(output['gry_shd']),np.float32)
        good=np.isfinite(raw)&(raw>0)
        reconstruction=output['gry_alb']*raw[:,:,None]
        original=output['lin_img']; valid_rgb=good[:,:,None]&np.isfinite(reconstruction)&np.isfinite(original)
        residual=float(np.max(np.abs(reconstruction-original)[valid_rgb])) if valid_rgb.any() else None
        if residual is None or residual>1e-5: raise RuntimeError('Inverse shading unit/reconstruction check failed')
        # Decode first; invalid support is conservatively tracked through resizing.
        sr=cv2.resize(np.where(good,raw,0),(512,512),interpolation=cv2.INTER_LINEAR)
        sg=(cv2.resize(good.astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)>=1)&np.isfinite(sr)&(sr>0)
        nonblack=np.any(rgb>0,axis=2); median_support=sg&nonblack
        if not median_support.any(): raise RuntimeError('No valid shading median support')
        median=float(np.median(sr[median_support])); rel=sr/median
        s0=np.where(sg,np.clip(rel,.25,4),1).astype(np.float32)
        n,nv=normal_inference(normal,proxy)
        mirrored,mv=normal_inference(normal,proxy[:,::-1].copy())
        mirrored=mirrored[:,::-1].copy(); mirrored[:,:,0]*=-1
        mirror_valid=nv&mv[:,::-1]&nonblack
        flip_ae=np.degrees(np.arccos(np.clip((n*mirrored).sum(2),-1,1)))
        flip_mean=float(flip_ae[mirror_valid].mean()) if mirror_valid.any() else None
        valid=sg&nv
        np.savez_compressed(ROOT/'cache_qc'/f'{scene}.npz',s_raw=sr,s0=s0,normal=n,valid_mask=valid,
                            shading_valid=sg,normal_valid=nv,risk_dark=(np.max(proxy,axis=2)<.01),
                            risk_sensor_saturation_unknown=np.ones((512,512),bool))
        vis=preview(scene,w,s0,n,valid,pool)
        info=dict(scene_id=scene,input_sha256=row['sha256'],seconds=time.time()-t,
          proxy_scale=scale,proxy_clip_fraction=0.,proxy_domain='camera-linear scaled; no validated color matrix',
          normal_basis='DSINE right/down/front; +Z frontal response, no sign flip',intrinsics='approximate FOV60 on512 square',
          shading_median=median,shading_invalid_fraction=float(1-sg.mean()),normal_invalid_fraction=float(1-nv.mean()),
          lower_clip_fraction=float((rel[median_support]<.25).mean()),upper_clip_fraction=float((rel[median_support]>4).mean()),
          reconstruction_max_abs=residual,normal_flip_equivariance_mean_deg=flip_mean,
          flip_note='New mirrored observation: reverse image positions and negate normal x. Diagnostic only, no calibrated threshold.',
          preview=vis,cache_sha256=sha(ROOT/'cache_qc'/f'{scene}.npz'))
        save(ROOT/'qc'/f'{scene}.json',info); summary.append(info)
        save(ROOT/'qc_summary.json',dict(status='IN_PROGRESS',count=len(summary),scenes=summary))
        print(json.dumps(dict(scene=scene,seconds=info['seconds'],invalid=float(1-valid.mean()))),flush=True)
    save(ROOT/'qc_summary.json',dict(status='INFERENCE_COMPLETE_VISUAL_REVIEW_PENDING',count=len(summary),seconds=time.time()-start,
         peak_gpu_bytes=torch.cuda.max_memory_allocated(),scenes=summary))

if __name__=='__main__': main()

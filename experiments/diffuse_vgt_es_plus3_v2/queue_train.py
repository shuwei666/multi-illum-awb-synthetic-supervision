"""Bounded seven-arm queue. Independent approvals are supplied, never generated."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from seven_synthesis import ARMS, Adapter

ROOT = Path(__file__).resolve().parent
PARENT = Path(os.environ.get('DIFFUSE_PARENT', str(ROOT.parent / 'diffuse_vgt_v1'))).resolve()
SPEC = Path(os.environ.get('ES_SPEC_DIR', str(ROOT/'specification' if (ROOT/'specification').exists() else PARENT/'repository/experiments/diffuse_vgt_es_plus3_v2'))).resolve()
FROZEN = ROOT / 'frozen'
TIE = ('O', 'ES00', 'ES10', 'ES01', 'ES11', 'KEEP_OLD_R', 'SHAM_OLD_0', 'NO_NEW_S')

def module(name):
    spec = importlib.util.spec_from_file_location(name, PARENT / 'train_arms.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

sys.path.insert(0, str(PARENT))
parent = module('private_parent')
engine = module('seven_engine')
read, write, sha, require = parent.read, parent.write, parent.sha, parent.require
_runtime = None

def runtime():
    global _runtime
    if _runtime is None:
        dg, synth = parent.runtime()
        _runtime = dg, Adapter(synth)
    return _runtime

def code_hashes():
    return {**parent.code_hashes(), **{p.name: sha(p) for p in (ROOT/'queue_train.py', ROOT/'seven_synthesis.py', ROOT/'test_seven_synthesis.py')}}

def check_frozen(check_sources=False):
    parent.check_frozen(check_sources=check_sources)
    frozen = read(FROZEN/'manifest.json')
    require(frozen['code_hashes'] == code_hashes(), 'Seven code changed after freeze')
    require(frozen['parent_manifest_sha256'] == sha(parent.FROZEN/'manifest.json'), 'Parent freeze changed')
    require(frozen['plan_sha256'] == sha(SPEC/'queue_plan.json') and frozen['spec_sha256'] == sha(SPEC/'GOAL_TASK.md'), 'Normative contract changed')
    require(frozen['arms'] == list(ARMS) and frozen['steps'] == 69600, 'Queue budget changed')
    for name, digest in frozen['copies'].items():
        require(sha(FROZEN/name) == digest == sha(parent.FROZEN/name), 'Frozen input copy changed')
    return frozen

def gate(scope, arm):
    p = FROZEN/f'{arm}_{scope}_approval.json'
    item = read(p)
    require(item['status'] == 'ADEQUATE' and item['scope'] == scope and item['arm'] == arm and item['freeze_manifest_sha256'] == sha(FROZEN/'manifest.json') and item['code_hashes'] == code_hashes(), 'Independent gate missing/stale')
    report = ROOT/item['independent_review']['path']
    require(report.resolve().is_relative_to(ROOT.resolve()) and sha(report) == item['independent_review']['sha256'], 'Independent review changed')
    review = read(report)
    require(review['artifact_verdict'] == 'PASS' and review['contract_verdict'] == 'ADEQUATE' and review['scope'] == scope and review['arm'] == arm and review['freeze_manifest_sha256'] == sha(FROZEN/'manifest.json') and review['code_hashes'] == code_hashes(), 'Independent review not adequate')
    for name in ('preflight', 'prior_calibration'):
        evidence = ROOT/item[name]['path']
        require(evidence.resolve().is_relative_to(ROOT.resolve()) and sha(evidence) == item[name]['sha256'] == review[name+'_sha256'], 'Evidence changed: '+name)
        require(read(evidence)['status'] == 'PASS', 'Evidence not PASS: '+name)
    require(read(ROOT/item['preflight']['path'])['freeze_manifest_sha256'] == sha(FROZEN/'manifest.json'), 'Preflight stale')
    if scope == 'formal':
        evidence = ROOT/item['calibration']['path']
        require(evidence.resolve().is_relative_to(ROOT.resolve()) and sha(evidence) == item['calibration']['sha256'] == review['calibration_sha256'], 'Calibration changed')
        engine.check_calibration(evidence, arm)
    return item

def state(arm=None, status=None, **extra):
    path = ROOT/'queue_state.json'
    data = read(path)
    if arm:
        data['arms'][arm].update(status=status, updated=time.time(), **extra)
        write(path, data)
    return data

def freeze():
    parent.check_frozen(check_sources=True)
    require(not FROZEN.exists(), 'Already frozen; use check')
    plan = read(SPEC/'queue_plan.json')
    require(plan['queue'] == list(ARMS) and plan['max_total_successful_updates'] == 487200, 'Unexpected queue')
    FROZEN.mkdir()
    for name in ('initial_seed0.pt', 'sources.json', 'endpoint_manifest.json'):
        shutil.copyfile(parent.FROZEN/name, FROZEN/name)
    manifest = dict(status='frozen', seed=0, steps=69600, arms=list(ARMS), config=parent.CONFIG, code_hashes=code_hashes(), parent_manifest_sha256=sha(parent.FROZEN/'manifest.json'), plan_sha256=sha(SPEC/'queue_plan.json'), spec_sha256=sha(SPEC/'GOAL_TASK.md'), copies={p.name:sha(p) for p in FROZEN.iterdir()}, reference_means=read(parent.FROZEN/'manifest.json')['reference_means'], created=time.time(), exact_resume='unsupported; interrupted formal arm is BLOCKED without restart')
    write(FROZEN/'manifest.json', manifest)
    write(ROOT/'frozen_config.json', dict(manifest, queue_plan=plan))
    write(ROOT/'asset_manifest.json', dict(parent_assets=parent.authorities(), parent_root=str(PARENT), initial_sha256=sha(FROZEN/'initial_seed0.pt')))
    write(ROOT/'queue_state.json', dict(queue=list(ARMS), max_successful_updates=487200, arms={a:dict(status='PLANNED', successful_updates=0) for a in ARMS}))

def tensor_sha(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def preflight():
    import torch
    check_frozen(True)
    require(not (ROOT/'preflight.json').exists() and not (ROOT/'calibration').exists() and not (ROOT/'runs').exists(), 'Preflight may not reset an existing batch')
    dg, synth = runtime()
    source, pool, bank, cache, basis = parent.load_inputs()
    cases = {}
    for cycle in (0, 416):
        entries = {}
        for arm in ARMS:
            capture = {}
            rendered = synth.render_cycle(source,pool,bank,parent.CONFIG,cycle,0,parent.budget(cycle),cache=cache,basis=basis,mode=arm,capture=capture)
            parent.validate_render(rendered,'B',parent.budget(cycle))
            original = capture['parent']['branch'][capture['parent']['order']] == 0
            original_patches = original.repeat_interleave(64)
            inherited = capture['parent']
            ids = inherited['order'][original]
            geom = tuple(g[ids] for g in inherited['geom'])
            frame = inherited['frame'][ids]
            expected_images = dg.base.zscore_bc(dg.base.rot_flip(frame,*geom).float())
            pp = dg.base.extract_patches(frame)
            pp = dg.base.rot_flip(pp.reshape(-1,3,16,16),*(g.repeat_interleave(256) for g in geom)).reshape(len(ids),256,3,16,16)
            expected_pp = torch.gather(pp,1,inherited['patches'][ids][...,None,None,None].expand(-1,-1,3,16,16)).reshape(-1,3,16,16)
            labels = synth.patch_mean(inherited['light'][ids].permute(0,3,1,2))
            labels = torch.gather(labels,1,inherited['patches'][ids][...,None].expand(-1,-1,3)).reshape(-1,3)
            labels /= labels.norm(dim=1,keepdim=True).clamp_min(dg.base.EPS)
            require(torch.equal(rendered[0][original],expected_images) and torch.equal(rendered[1][original_patches],expected_pp) and torch.equal(rendered[2][original_patches],labels),'Original branch differs from inherited O')
            identity = [tensor_sha(rendered[0][original]), tensor_sha(rendered[1][original_patches]), tensor_sha(rendered[2][original_patches])]
            entries[arm] = dict(tensors=[tensor_sha(t) for t in rendered[:5]], original_identity=identity, sampling=rendered[5]['matched_sampling_sha256'], valid_patches=rendered[5]['consumed_valid_patches'], ledger={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in rendered[6].items()}, diag=rendered[5])
            del rendered, capture, pp, expected_images, expected_pp, labels, frame
        require(len({e['sampling'] for e in entries.values()}) == 1, 'Full/tail draws/mask mismatch')
        require(len({json.dumps(e['ledger'],sort_keys=True) for e in entries.values()}) == 1, 'Endpoint consumption mismatch')
        require(len({json.dumps(e['original_identity']) for e in entries.values()}) == 1, 'Original images/patches/GT changed across arms')
        for group in (('ES00','ES01','SHAM_OLD_0','NO_NEW_S'),('ES10','ES11','KEEP_OLD_R')):
            require(len({entries[a]['tensors'][2] for a in group}) == 1, 'Actual label mismatch')
        require(len({e['tensors'][3] for e in entries.values()}) == 1 and len({e['tensors'][4] for e in entries.values()}) == 1, 'Actual mask/index mismatch')
        cases[str(cycle)] = entries
    write(ROOT/'preflight.json',dict(status='PASS',scope='actual_full_tail_pipeline_no_optimizer',freeze_manifest_sha256=sha(FROZEN/'manifest.json'),cases=cases))
    for arm in ARMS: state(arm,'READY')

# Reuse reviewed optimizer/budget/ledger machinery, keeping parent's globals intact.
engine.ROOT, engine.FROZEN, engine.ARMS, engine.ORDER = ROOT, FROZEN, ARMS, TIE
engine.runtime, engine.code_hashes, engine.check_frozen, engine.check_gate = runtime, code_hashes, check_frozen, gate
engine.load_inputs, engine.fresh_model = parent.load_inputs, parent.fresh_model
engine.validate_render = lambda rendered, arm, steps: parent.validate_render(rendered,'B',steps)
def select_winner(scores):
    import math
    require(set(scores) == set(TIE) and all(math.isfinite(x) and x > 0 for x in scores.values()), 'Invalid candidate scores')
    return min(TIE, key=scores.get)
engine.select_winner = select_winner

def run(arm, calibration):
    check_frozen(True)
    if calibration:
        require(not (ROOT/'calibration'/'entry'/arm).exists(), 'Calibration budget already reserved; cannot repeat')
    else:
        require(state()['arms'][arm]['status'] == 'READY', 'Arm not READY; no unsafe restart')
        earlier = ARMS[:ARMS.index(arm)]
        require(all(state()['arms'][a]['status']=='COMPLETE' for a in earlier), 'Queue order violated')
        state(arm,'RUNNING')
    try:
        engine.run(arm,calibration=calibration,name='entry' if calibration else None)
        if not calibration:
            engine.checked_run(arm)
            state(arm,'COMPLETE',successful_updates=69600,evidence=f'runs/{arm}_seed0/completion.json')
    except Exception as exc:
        if not calibration:
            failure=ROOT/'runs'/f'{arm}_seed0'/'failure.json'
            state(arm,'FAILED',successful_updates=read(failure)['successful_updates'] if failure.exists() else 0,error=repr(exc))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('freeze','check','status','preflight','calibrate','train','queue','dev'))
    p.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'Optimized Python prohibited')
    if args.mode=='status': print(json.dumps(state(),indent=2)); return
    if args.mode=='freeze': freeze(); return
    if args.mode=='check': check_frozen(True); return
    dg,_=runtime()
    with dg.r3.exclusive_lock():
        dg.r3.gpu_preflight()
        if args.mode=='preflight': preflight()
        elif args.mode=='dev':
            engine.dev()
            shutil.copyfile(ROOT/'development_selection.json',ROOT/'selection_before_real_eval.json')
        elif args.mode=='queue':
            for arm in ARMS:
                if state()['arms'][arm]['status']=='COMPLETE': engine.checked_run(arm); continue
                run(arm,False)
        else:
            require(args.arm is not None,'Explicit arm required')
            run(args.arm,args.mode=='calibrate')

if __name__=='__main__': main()

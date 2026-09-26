"""Gated diffuse A/B/C runner. No mixed evaluation and no automatic resume.

Cache manifest schema 1: status=complete, source_manifest_sha256,
priors_manifest_sha256, cache_code_sha256,
normal_basis (3x3 rows x/y/front), unusable_source_fraction, and files mapping
scene id -> {path: cache_all/<scene>.npz, sha256, input_sha256}.
Approval files are written by an independent reviewer, never this program.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'frozen'
ARMS = ('A', 'B', 'C')
ORDER = ('O', *ARMS)
STEPS = 69600
CONFIG = {'original_half': True, 'bias': False, 'endpoints': 'full'}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def runtime():
    sys.path.insert(0, str(ROOT.parent / 'code'))
    import dg_experiment as dg
    import diffuse_synthesis as synth
    return dg, synth

def code_hashes():
    dg, synth = runtime()
    return {**dg.code_hashes(), **{p.name: sha(p) for p in
            (Path(__file__), Path(synth.__file__), ROOT / 'render_diffuse.py', ROOT / 'cache_priors.py')}}

def select_winner(scores):
    require(tuple(scores) == ORDER, 'Candidate inventory/tie order changed')
    require(all(math.isfinite(x) and x > 0 for x in scores.values()), 'Invalid scores')
    return min(ORDER, key=scores.get)

def budget(cycle):
    require(type(cycle) is int and 0 <= cycle < 417, 'Cycle outside bounded batch')
    return min(167, STEPS - 167 * cycle)

def check_cache():
    """Reject QC-only, missing, partial, stale, or substituted prior caches."""
    manifest = read(ROOT / 'cache_manifest.json')
    source = read(ROOT / 'source_manifest.json')
    require(manifest['schema_version'] == 1 and manifest['status'] == 'complete', 'Incomplete cache')
    require(manifest['source_manifest_sha256'] == sha(ROOT / 'source_manifest.json'), 'Stale cache sources')
    require(manifest.get('priors_manifest_sha256') == sha(ROOT / 'priors_manifest.json'), 'Stale/missing cache prior provenance')
    require(manifest.get('cache_code_sha256') == sha(ROOT / 'cache_priors.py'), 'Stale/missing cache code provenance')
    rows = source['sources']
    ids = [r['scene_id'] for r in rows]
    require(len(ids) == len(set(ids)) == source['source_count'] == 668, 'Source inventory changed')
    parent = read(ROOT.parent / 'dg_matched/frozen/sources.json')
    require(ids == parent['source_ids'] and {r['scene_id']: r['sha256'] for r in rows} == parent['raw_sha256'], 'Cache sources differ from frozen parent')
    require(set(manifest['files']) == set(ids), 'Full source cache required')
    fraction = manifest['unusable_source_fraction']
    require(math.isfinite(fraction) and 0 <= fraction <= .2, 'Too many unusable sources')
    basis = manifest['normal_basis']
    require(len(basis) == 3 and all(len(row) == 3 for row in basis), 'Missing reviewed basis')
    for i in range(3):
        for j in range(3):
            dot = sum(basis[i][k] * basis[j][k] for k in range(3))
            require(math.isfinite(dot) and abs(dot - (i == j)) < 1e-5, 'Invalid normal basis')
    for row in rows:
        item = manifest['files'][row['scene_id']]
        path = (ROOT / item['path']).resolve(strict=True)
        require(path.parent == (ROOT / 'cache_all').resolve() and path.name == row['scene_id'] + '.npz', 'Cache path outside all-source inventory')
        require(item['input_sha256'] == row['sha256'] and sha(path) == item['sha256'], 'Cache/input hash mismatch')
    return manifest

def authorities():
    dg, _ = runtime()
    dg.check_frozen(check_sources=False)
    paths = [ROOT / name for name in ('USER_TASK.txt', 'REVISION_1.md', 'source_manifest.json',
             'cache_manifest.json', 'priors_manifest.json', 'source_access_policy.json')]
    paths += [dg.FROZEN / name for name in ('manifest.json', 'initial_seed0.pt', 'sources.json', 'endpoint_manifest.json')]
    paths += dg.authority_paths()
    return {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}

def freeze():
    cache = check_cache()
    assets = authorities()
    require(not FROZEN.exists(), 'Already frozen')
    dg, _ = runtime()
    staging = Path(tempfile.mkdtemp(prefix='freeze-staging-', dir=ROOT))
    for name in ('initial_seed0.pt', 'sources.json', 'endpoint_manifest.json'):
        shutil.copyfile(dg.FROZEN / name, staging / name)
    write(staging / 'manifest.json', dict(status='frozen', seed=0, steps=STEPS, arms=list(ARMS),
          config=CONFIG, code_hashes=code_hashes(), assets=assets,
          copies={p.name: sha(p) for p in staging.iterdir()},
          reference_means=read(dg.R2 / 'dev_results/M.json')['means'], created=time.time()))
    staging.rename(FROZEN)

def check_frozen(check_sources=False):
    frozen = read(FROZEN / 'manifest.json')
    require(frozen['status'] == 'frozen' and frozen['seed'] == 0 and frozen['steps'] == STEPS
            and frozen['arms'] == list(ARMS) and frozen['config'] == CONFIG, 'Frozen contract changed')
    require(frozen['code_hashes'] == code_hashes() and frozen['assets'] == authorities(), 'Frozen code/assets changed')
    dg, _ = runtime()
    require(set(frozen['copies']) == {'initial_seed0.pt', 'sources.json', 'endpoint_manifest.json'}, 'Frozen copies missing')
    for name, digest in frozen['copies'].items():
        require(sha(FROZEN / name) == digest == sha(dg.FROZEN / name), 'Frozen copy changed')
    require(sha(FROZEN / 'initial_seed0.pt') == 'a81e752797ae353ce8e8081013405906f9d8975a6f5899c9150df2806033336a', 'Wrong initialization')
    check_cache()
    if check_sources:
        _, _, source = dg.r3.source_inventory()
        require(source == read(FROZEN / 'sources.json'), 'Live source changed')
    return frozen

def evidence(item):
    path = (ROOT / item['path']).resolve(strict=True)
    require(path.is_relative_to(ROOT.resolve()) and sha(path) == item['sha256'], 'Missing/stale local evidence')
    return path

def check_gate(scope, arm):
    require(scope in ('calibration', 'formal') and arm in ARMS, 'Unknown entry')
    gate = read(FROZEN / f'{arm}_{scope}_approval.json')
    digest = sha(FROZEN / 'manifest.json')
    require(gate['status'] == 'ADEQUATE' and gate['scope'] == scope and gate['arm'] == arm
            and gate['freeze_manifest_sha256'] == digest and gate['code_hashes'] == code_hashes(), 'Entry gate missing/stale')
    report = read(evidence(gate['independent_review']))
    require(report['artifact_verdict'] == 'PASS' and report['contract_verdict'] == 'ADEQUATE'
            and report['scope'] == scope and report['arm'] == arm
            and report['freeze_manifest_sha256'] == digest and report['code_hashes'] == code_hashes(), 'Independent approval missing/stale')
    for key in ('real24_qc', 'matched_preflight', 'input_sensitivity'):
        path = evidence(gate[key])
        require(report[key + '_sha256'] == sha(path), 'Unreviewed entry evidence')
        item = read(path)
        require(item['status'] == 'PASS' and item['freeze_manifest_sha256'] == digest, 'Failed/stale evidence')
    if scope == 'formal':
        done_path = evidence(gate['calibration'])
        require(report['calibration_sha256'] == sha(done_path), 'Unreviewed calibration')
        check_calibration(done_path, arm)
    return gate

def check_calibration(path, arm):
    done = read(path)
    require(done['status'] == 'calibration_complete' and done['arm'] == arm
            and done['freeze_manifest_sha256'] == sha(FROZEN / 'manifest.json'), 'Calibration identity changed')
    require(set(done['cases']) == {'full', 'tail'}, 'Full/tail calibration required')
    for name, expected_cycle in (('full', 0), ('tail', 416)):
        case = done['cases'][name]
        require(case['steps'] == 1 and case['cycle'] == expected_cycle and case['fresh_initialization'] is True,
                'Calibration must use separate fresh weights')
        require(case['initial_sha256'] == sha(FROZEN / 'initial_seed0.pt'), 'Calibration initialization changed')
        require(all(math.isfinite(case[k]) and case[k] > 0 for k in ('grad_norm', 'parameter_delta')), 'Invalid calibration update')
        expected_index = 0 if name == 'full' else 127
        require(case['batch_indices'] == [expected_index] and case['consumed_views'] == 8, 'Calibration consumption mismatch')
    for name, digest in done['outputs'].items():
        require(Path(name).name == name and sha(path.parent / name) == digest, 'Calibration output changed')
    require(set(done['outputs']) == {'manifest.json', 'history.jsonl', 'recovery.pt', 'full.pt', 'tail.pt'}, 'Incomplete calibration outputs')
    history = [json.loads(line) for line in (path.parent / 'history.jsonl').read_text().splitlines()]
    require(len(history) == 2, 'Calibration history incomplete')
    for row, index in zip(history, (0, 127)):
        diag = row['source_render_diag']
        require(row['batch_indices'] == diag['consumed_batch_indices'] == [index] and row['consumed_views'] == diag['consumed_views'] == 8, 'Calibration diagnostic consumption mismatch')
        require(len(row['updates']) == 1 and row['valid_patches'] == diag['consumed_valid_patches'] == row['updates'][0]['valid_patches'], 'Calibration valid patch mass mismatch')
        require(0 <= diag['consumed_mixed_valid_patches'] <= row['valid_patches'] <= 512 and 0 <= diag['consumed_mixed_views'] <= 8, 'Calibration mass outside one batch')

def load_inputs():
    import numpy as np
    import torch
    dg, synth = runtime()
    rows, pool_rgb, sources = dg.r3.source_inventory()
    require(sources == read(FROZEN / 'sources.json'), 'Source order changed')
    manifest = check_cache()
    fields = {'s0': [], 'normal': [], 'shading_valid': [], 'normal_valid': []}
    for scene, _, _ in rows:
        with np.load(ROOT / manifest['files'][scene]['path'], allow_pickle=False) as arr:
            s, n, v = arr['s0'], arr['normal'], arr['valid_mask']
            sv, nv = arr['shading_valid'], arr['normal_valid']
            require(s.shape == v.shape == (512, 512) and n.shape == (512, 512, 3), 'Prior shape changed')
            require(s.dtype == n.dtype == np.float32 and v.dtype == np.bool_, 'Prior dtype changed')
            require(sv.shape == nv.shape == v.shape and sv.dtype == nv.dtype == np.bool_ and np.array_equal(v, sv & nv), 'Separate cache validity changed')
            require(np.isfinite(s).all() and np.isfinite(n).all() and (s >= .25).all() and (s <= 4).all(), 'Nonfinite/unprotected priors')
            fields['s0'].append(s[None]); fields['normal'].append(n.transpose(2, 0, 1))
            fields['shading_valid'].append(sv[None]); fields['normal_valid'].append(nv[None])
    cache = {k: torch.from_numpy(np.stack(v)).cuda() for k, v in fields.items()}
    source = dg.r2.load_native(rows, 'cuda')
    pool = torch.from_numpy(pool_rgb).cuda()
    bank, endpoints = synth.load_pair_bank(dg.DATA, pool_rgb, 'cuda') if hasattr(synth, 'load_pair_bank') else dg.synth.load_pair_bank(dg.DATA, pool_rgb, 'cuda')
    require(dg.r3.json_value(endpoints) == read(FROZEN / 'endpoint_manifest.json'), 'Endpoint changed')
    return source, pool, bank, cache, torch.tensor(manifest['normal_basis'], dtype=torch.float32, device='cuda')

def fresh_model():
    import torch
    dg, _ = runtime()
    dg.base.seed_all(0)
    model = dg.base.DomislovicNet().cuda()
    require(sum(p.numel() for p in model.parameters()) == 42179, 'Consumer changed')
    model.load_state_dict(torch.load(FROZEN / 'initial_seed0.pt', map_location='cuda', weights_only=True))
    model.train()
    return model, torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=5e-5)

def update(model, optimizer, rendered, batch_index, global_step):
    import torch
    dg, _ = runtime()
    images, patches, gt, valid, idx = rendered[:5]
    sl = slice(batch_index * 512, (batch_index + 1) * 512)
    keep = valid[sl]
    require(bool(keep.any()), 'Empty batch cannot count as successful update')
    unique, inverse = torch.unique(idx[sl], return_inverse=True)
    pred = model(model.encode_image(images[unique]), dg.base.zscore_bc(patches[sl].float()), inverse)
    loss = dg.base.eq5_loss(pred[keep], gt[sl][keep])
    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    for group in optimizer.param_groups:
        group['lr'] = dg.base.triangular_lr(global_step / 174)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite gradients')
    norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite parameters')
    delta = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(model.parameters(), before)))
    require(norm > 0 and delta > 0, 'No effective update')
    return dict(loss=float(loss), grad_norm=norm, parameter_delta=delta, valid_patches=int(keep.sum()), candidate_patches=512)

def validate_render(rendered, arm, steps):
    import numpy as np
    import torch
    dg, _ = runtime()
    images, patches, gt, valid, idx, diag, ledger = rendered
    require(len(images) == 1336 and len(patches) == len(gt) == len(valid) == len(idx) == 85504, 'View/patch inventory changed')
    require(diag['mode'] == arm and diag['branch_counts'] == [668, 0, 668] and diag['common_arm_mask_verified'] is True, 'Construction changed')
    require(len(diag['matched_sampling_sha256']) == 64, 'Missing matched sampling digest')
    require(diag['consumed_views'] == steps * 8 and len(diag['consumed_batch_indices']) == steps
            and diag['consumed_valid_patches'] == int(valid.reshape(-1, 512)[diag['consumed_batch_indices']].sum()), 'Actual optimizer consumption mismatch')
    require(all(bool(torch.isfinite(t).all()) for t in (images, patches, gt)) and bool((gt > 0).all()), 'Nonfinite render')
    require(set(ledger) == set(dg.r3.LEDGER_KEYS), 'Ledger inventory changed')
    for key, count in ledger.items():
        require(count.shape == (1353,) and count.dtype == np.int64 and (count >= 0).all(), 'Invalid ledger')
        mass = diag['consumed_mixed_valid_patches' if 'patch' in key else 'consumed_mixed_views']
        require(int(count.sum()) == mass and 0 <= mass <= steps * (512 if 'patch' in key else 8), 'Invalid ledger mass')

def recovery(out, model, optimizer, step, cycle, batch_index, ledgers):
    import numpy as np
    import torch
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
        cycle=cycle, next_batch_index=batch_index, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
        rng_numpy=np.random.get_state(), rng_python=random.getstate(), ledgers=ledgers,
        freeze_manifest_sha256=sha(FROZEN / 'manifest.json'), resume='requires independent review; not automatic'), out / 'recovery.tmp')
    os.replace(out / 'recovery.tmp', out / 'recovery.pt')

def run(arm, calibration=False, name=None):
    import numpy as np
    import torch
    require(arm in ARMS, 'One explicit arm required')
    frozen = check_frozen(check_sources=True)
    scope = 'calibration' if calibration else 'formal'
    check_gate(scope, arm)
    if calibration:
        require(name and Path(name).name == name and name not in ('.', '..'), 'Unique calibration name required')
    out = ROOT / 'calibration' / name / arm if calibration else ROOT / 'runs' / (arm + '_seed0')
    out.mkdir(parents=True, exist_ok=False)
    dg, synth = runtime()
    start = time.monotonic()
    digest = sha(FROZEN / 'manifest.json')
    manifest = dict(arm=arm, seed=0, steps=2 if calibration else STEPS, calibration=calibration,
        code_hashes=code_hashes(), freeze_manifest_sha256=digest, initial_sha256=sha(FROZEN / 'initial_seed0.pt'),
        approval_sha256=sha(FROZEN / f'{arm}_{scope}_approval.json'), config=CONFIG, real_mixed_gt_access='none')
    write(out / 'manifest.json', manifest)
    step = 0
    try:
        with dg.r3.heartbeat(out, arm=arm, stage='loading') as state:
            source, pool, bank, cache, basis = load_inputs()
            cases, ledgers = {}, {k: [] for k in dg.r3.LEDGER_KEYS}
            model, optimizer = fresh_model()
            cycles = (0, 416) if calibration else range(417)
            with (out / 'history.jsonl').open('x') as history:
                for cycle in cycles:
                    if calibration:
                        model, optimizer = fresh_model()
                    state.update(stage='synthesis', step=step, cycle=cycle)
                    cycle_started = time.monotonic()
                    indices = ((0 if cycle == 0 else budget(cycle) - 1),) if calibration else range(budget(cycle))
                    rendered = synth.render_cycle(source, pool, bank, CONFIG, cycle, 0, budget(cycle), cache=cache, basis=basis, mode=arm, batch_indices=list(indices) if calibration else None)
                    validate_render(rendered, arm, len(indices))
                    torch.cuda.synchronize()
                    synthesis_sec = time.monotonic() - cycle_started
                    records = []
                    state.update(stage='training')
                    for batch_index in indices:
                        stats = update(model, optimizer, rendered, batch_index, cycle * 167 + batch_index)
                        step += 1
                        records.append(stats)
                    record = dict(cycle=cycle, steps=step, consumed_views=len(records) * 8,
                        batch_indices=list(indices),
                        loss=float(np.mean([x['loss'] for x in records])), valid_patches=sum(x['valid_patches'] for x in records),
                        source_render_diag=rendered[5], updates=records, synthesis_sec=synthesis_sec,
                        online_seconds_per_update=(time.monotonic() - cycle_started) / len(records),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    history.write(json.dumps(record, allow_nan=False) + '\n'); history.flush()
                    for key in ledgers:
                        ledgers[key].append(rendered[6][key].copy())
                    recovery(out, model, optimizer, step, cycle, indices[-1] + 1, ledgers)
                    if calibration:
                        case = 'full' if cycle == 0 else 'tail'
                        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), out / (case + '.pt'))
                        cases[case] = dict(**stats, steps=1, cycle=cycle, fresh_initialization=True, initial_sha256=manifest['initial_sha256'], batch_indices=list(indices), consumed_views=8)
                    state.update(step=step, cycle=cycle)
                    del rendered
                require(step == (2 if calibration else STEPS), 'Successful budget mismatch')
            if not calibration:
                torch.save(model.state_dict(), out / 'ckpt_final.pt')
                np.savez_compressed(out / 'endpoint_counts.npz', **{k: np.stack(v) for k, v in ledgers.items()})
            outputs = ['manifest.json', 'history.jsonl', 'recovery.pt'] + (['full.pt', 'tail.pt'] if calibration else ['ckpt_final.pt', 'endpoint_counts.npz'])
            write(out / 'completion.json', dict(status='calibration_complete' if calibration else 'training_complete',
                arm=arm, steps=step, cycles=len(cycles), cases=cases, freeze_manifest_sha256=digest,
                seconds=time.monotonic() - start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                outputs={k: sha(out / k) for k in outputs}))
            state.update(status='complete')
    except Exception as exc:
        write(out / 'failure.json', dict(status='FAILED', successful_updates=step, error=repr(exc), freeze_manifest_sha256=digest))
        raise

def checked_run(arm):
    import numpy as np
    out = ROOT / 'runs' / (arm + '_seed0')
    done, manifest = read(out / 'completion.json'), read(out / 'manifest.json')
    require(done['status'] == 'training_complete' and done['steps'] == manifest['steps'] == STEPS and done['cycles'] == 417, 'Incomplete formal run')
    require(done['arm'] == manifest['arm'] == arm and manifest['seed'] == 0 and manifest['calibration'] is False, 'Wrong run identity')
    require(manifest['code_hashes'] == code_hashes() and manifest['config'] == CONFIG, 'Run code/config changed')
    require(done['freeze_manifest_sha256'] == manifest['freeze_manifest_sha256'] == sha(FROZEN / 'manifest.json'), 'Stale run')
    require(manifest['initial_sha256'] == sha(FROZEN / 'initial_seed0.pt'), 'Wrong initialization')
    require(set(done['outputs']) == {'manifest.json', 'history.jsonl', 'recovery.pt', 'ckpt_final.pt', 'endpoint_counts.npz'}, 'Missing outputs')
    for name, digest in done['outputs'].items():
        require(sha(out / name) == digest, 'Changed formal output')
    history = [json.loads(x) for x in (out / 'history.jsonl').read_text().splitlines()]
    require(len(history) == 417 and [x['steps'] for x in history] == [min((c + 1) * 167, STEPS) for c in range(417)], 'History budget mismatch')
    dg, _ = runtime()
    with np.load(out / 'endpoint_counts.npz', allow_pickle=False) as counts:
        require(set(counts.files) == set(dg.r3.LEDGER_KEYS), 'Endpoint inventory changed')
        for key in counts.files:
            values = counts[key]
            require(values.shape == (417, 1353) and values.dtype == np.int64 and (values >= 0).all(), 'Invalid endpoint ledger')
            for c, row in enumerate(history):
                require(row['cycle'] == c and len(row['updates']) == budget(c) and row['consumed_views'] == budget(c) * 8, 'Consumption budget changed')
                require(all(math.isfinite(x['loss']) and x['candidate_patches'] == 512 and 0 < x['valid_patches'] <= 512 for x in row['updates']), 'Invalid update history')
                diag = row['source_render_diag']
                mass = diag['consumed_mixed_valid_patches' if 'patch' in key else 'consumed_mixed_views']
                require(int(values[c].sum()) == mass and 0 <= mass <= budget(c) * (512 if 'patch' in key else 8), 'Endpoint mass changed')
    return out / 'ckpt_final.pt', history

def dev():
    import numpy as np
    import torch
    frozen = check_frozen(check_sources=True)
    dg, _ = runtime()
    runs = {arm: checked_run(arm) for arm in ARMS}
    for c in range(417):
        require(len({runs[a][1][c]['source_render_diag']['matched_sampling_sha256'] for a in ARMS}) == 1, 'Unmatched draw/mask history')
    ledgers = [np.load(ROOT / 'runs' / (a + '_seed0') / 'endpoint_counts.npz', allow_pickle=False) for a in ARMS]
    try:
        for key in dg.r3.LEDGER_KEYS:
            require(all(np.array_equal(ledgers[0][key], item[key]) for item in ledgers[1:]), 'Unmatched endpoint consumption')
    finally:
        for item in ledgers: item.close()
    target = ROOT / 'development_selection.json'
    require(not target.exists(), 'Selection exists')
    parent, parent_dev, parent_ckpt = dg.checked_parent()
    results, checkpoints = {'O': parent}, {'O': parent_ckpt}
    for arm in ARMS:
        ckpt = runs[arm][0]; checkpoints[arm] = ckpt
        model = dg.base.DomislovicNet().cuda()
        model.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True)); model.eval()
        means = {}
        for mode in dg.r2.MODES:
            data = torch.load(dg.R2 / 'frozen_dev' / (mode + '.pt'), map_location='cuda', weights_only=True)
            loaded = (data['frames'], data['gt'], data['valid'], torch.ones(len(data['frames']), device='cuda'))
            with torch.inference_mode():
                _, error, valid, _ = dg.r2.evaluation.predict(model, loaded, 'cuda')
            means[mode] = float(error[valid].mean())
            del data, loaded
        results[arm] = dict(means=means, score=dg.r3.dev_score(means, frozen['reference_means']), checkpoint_sha256=sha(ckpt))
        del model
    scores = {a: results[a]['score'] for a in ORDER}
    write(target, dict(status='development_selected_real_evaluation_not_run', winner=select_winner(scores),
        scores=scores, results=results, tie_order=list(ORDER), selected_at=time.time(),
        freeze_manifest_sha256=sha(FROZEN / 'manifest.json'), checkpoint_sha256={a: sha(p) for a, p in checkpoints.items()}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'check', 'calibrate', 'train', 'dev'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--name')
    parser.add_argument('--scope', choices=('calibration', 'formal'))
    args = parser.parse_args()
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Optimized Python unsupported')
    dg, _ = runtime()
    with dg.r3.exclusive_lock():
        if args.mode == 'freeze': freeze()
        elif args.mode == 'check':
            check_frozen(check_sources=True)
            if args.scope: check_gate(args.scope, args.arm)
        else:
            dg.r3.gpu_preflight()
            if args.mode == 'dev': dev()
            else: run(args.arm, calibration=args.mode == 'calibrate', name=args.name)

if __name__ == '__main__':
    main()

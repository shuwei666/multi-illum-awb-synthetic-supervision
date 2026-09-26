"""Independent CPU-only completed-cohort audit; never trains or reads real GT.

Run only after A/B/C and development_selection.json have completed. This script
does not grant approval: its JSON evidence must be reviewed before a separate
reviewer authors evaluation_approval.json. Shared disk/model: isolation limited.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import train_arms as r


def need(ok, msg):
    if not ok:
        raise AssertionError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit():
    need(not torch.cuda.is_initialized(), 'GPU initialized')
    for arm in r.ARMS:
        need((ROOT / 'runs' / (arm + '_seed0') / 'completion.json').is_file(), 'BLOCKED: incomplete ' + arm)
    need((ROOT / 'development_selection.json').is_file(), 'BLOCKED: development missing')
    need(not (ROOT / 'results').exists(), 'Real scoring already started; entry scope invalid')
    frozen = r.check_frozen(check_sources=False)
    dg, _ = r.runtime()
    initial = torch.load(r.FROZEN / 'initial_seed0.pt', map_location='cpu', weights_only=True)
    evidence, histories, ledgers = {}, {}, {}
    matched_keys = ('matched_sampling_sha256', 'branch_counts', 'valid_patches',
                    'parent_valid_patches', 'technical_rejected_patches',
                    'consumed_valid_patches', 'consumed_mixed_valid_patches',
                    'consumed_mixed_views', 'consumed_views', 'consumed_batch_indices',
                    'common_arm_mask_verified', 'invalid_fraction')
    for arm in r.ARMS:
        r.check_gate('formal', arm)
        checkpoint, history = r.checked_run(arm)
        out = checkpoint.parent
        need(not (out / 'failure.json').exists(), arm + ': recorded failure')
        manifest, done = read(out / 'manifest.json'), read(out / 'completion.json')
        need(manifest['approval_sha256'] == sha(r.FROZEN / (arm + '_formal_approval.json')), 'Run approval differs')
        need(manifest['real_mixed_gt_access'] == 'none', 'GT access declaration differs')
        for name, digest in done['outputs'].items():
            need(sha(out / name) == digest, arm + ': completion hash ' + name)
        final = torch.load(checkpoint, map_location='cpu', weights_only=True)
        # Authorized own recovery contains numpy RNG state, hence weights_only=False.
        recovery = torch.load(out / 'recovery.pt', map_location='cpu', weights_only=False)
        need(finite_tree(final) and finite_tree(recovery['model']) and finite_tree(recovery['optimizer']), 'Nonfinite state')
        need(set(final) == set(initial) == set(recovery['model']), 'Model key mismatch')
        need(all(torch.equal(final[k], recovery['model'][k]) for k in final), 'Recovery/checkpoint differ')
        delta = sum(float((final[k].double() - initial[k].double()).square().sum()) for k in final) ** .5
        need(math.isfinite(delta) and delta > 0, 'Final equals initialization')
        need(recovery['step'] == 69600 and recovery['cycle'] == 416 and recovery['next_batch_index'] == 128, 'Recovery budget mismatch')
        need(recovery['freeze_manifest_sha256'] == sha(r.FROZEN / 'manifest.json'), 'Recovery freeze mismatch')
        optimizer = recovery['optimizer']
        params = [p for group in optimizer['param_groups'] for p in group['params']]
        need(len(params) == len(set(params)) and set(params) == set(optimizer['state']), 'Missing Adam states')
        need(len(params) == len(list(dg.base.DomislovicNet().parameters())), 'Adam parameter inventory')
        for state in optimizer['state'].values():
            need(float(state['step']) == 69600, 'Adam step not 69600')
            need(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam state schema')
            need(bool((state['exp_avg_sq'] >= 0).all()), 'Negative Adam variance')
        with np.load(out / 'endpoint_counts.npz', allow_pickle=False) as arr:
            ledger = {k: arr[k].copy() for k in arr.files}
        need(set(ledger) == set(recovery['ledgers']) == set(dg.r3.LEDGER_KEYS), 'Ledger schema')
        for k in ledger:
            need(np.array_equal(ledger[k], np.stack(recovery['ledgers'][k])), 'Recovery ledger differs')
        for c, row in enumerate(history):
            count = 167 if c < 416 else 128
            diag = row['source_render_diag']
            need(row['cycle'] == c and row['batch_indices'] == diag['consumed_batch_indices'] == list(range(count)), 'Batch consumption')
            need(row['consumed_views'] == diag['consumed_views'] == count * 8, 'View consumption')
            need(len(row['updates']) == count and row['steps'] == min((c + 1) * 167, 69600), 'Successful updates')
            need(row['valid_patches'] == diag['consumed_valid_patches'] == sum(u['valid_patches'] for u in row['updates']), 'Valid loss mass')
            need(diag['mode'] == arm and diag['branch_counts'] == [668, 0, 668] and diag['common_arm_mask_verified'] is True, 'Branch/mask changed')
            need(finite_tree(row), 'Nonfinite history')
            for u in row['updates']:
                need(u['grad_norm'] > 0 and u['parameter_delta'] > 0 and u['candidate_patches'] == 512 and 0 < u['valid_patches'] <= 512, 'Ineffective update')
            need(abs(row['loss'] - np.mean([u['loss'] for u in row['updates']])) < 1e-12, 'Mean loss mismatch')
        histories[arm], ledgers[arm] = history, ledger
        evidence[arm] = dict(parameter_delta_from_initial=delta, adam_parameter_states=len(params),
                             adam_steps=69600, cycles=417, completion_sha256=sha(out / 'completion.json'),
                             recovery_sha256=sha(out / 'recovery.pt'), checkpoint_sha256=sha(checkpoint),
                             seconds=done['seconds'], peak_gpu_bytes=done['peak_gpu_bytes'])
    for arm in ('B', 'C'):
        for c in range(417):
            for key in matched_keys:
                need(histories['A'][c]['source_render_diag'][key] == histories[arm][c]['source_render_diag'][key], 'Unmatched ' + key)
        for k in ledgers['A']:
            need(np.array_equal(ledgers['A'][k], ledgers[arm][k]), 'Unmatched endpoint arrays')
    spec = importlib.util.spec_from_file_location('diffuse_scoring_audited', ROOT / 'evaluate_frozen.py')
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    selection, checkpoints = scoring.checked_selection(r, dg)
    need(frozen['created'] < selection['selected_at'] <= time.time(), 'Selection time invalid')
    for arm in r.ORDER:
        means = selection['results'][arm]['means']
        ratios = {k: means[k] / frozen['reference_means'][k] for k in means}
        independent_score = .5 * ratios['real_single'] + .125 * sum(ratios[k] for k in ('global', 'sigmoid', 'blob', 'lowfreq'))
        need(math.isclose(independent_score, selection['scores'][arm], rel_tol=1e-14), 'Independent dev formula')
    need(selection['winner'] == min(r.ORDER, key=lambda a: selection['scores'][a]), 'Frozen winner/tie order')
    baseline = ROOT.parent / 'evidence/baseline_nikon_seed0_ckpt_final.pt'
    need(sha(baseline) == scoring.BASELINE_SHA, 'Baseline hash')
    reference = dg.R2 / 'final_eval/evaluation_manifest.json'
    need(sha(reference) == scoring.REFERENCE_SHA, 'Reference manifest hash')
    checkpoints = {'baseline': baseline, **checkpoints}
    bindings = dict(scope='diffuse_final_evaluation', freeze_manifest_sha256=sha(r.FROZEN / 'manifest.json'),
        development_selection_sha256=sha(ROOT / 'development_selection.json'),
        checkpoint_sha256={a: sha(p) for a, p in checkpoints.items()},
        completion_evidence_sha256={a: {name: sha(ROOT / 'runs' / (a + '_seed0') / name)
            for name in ('completion.json', 'history.jsonl', 'manifest.json', 'endpoint_counts.npz')} for a in r.ARMS},
        code_hashes=r.code_hashes(), implementation_hashes=scoring.implementation_hashes(),
        official_reference_manifest_sha256=scoring.REFERENCE_SHA)
    need(not torch.cuda.is_initialized(), 'GPU initialized during audit')
    return dict(status='CPU_AUDIT_PASS_NOT_APPROVAL', reviewed_at=time.time(), isolation='limited: shared model/disk',
                real_mixed_gt_access='none', audit_script_sha256=sha(__file__), arms=evidence,
                selection=selection, bindings=bindings)


if __name__ == '__main__':
    result = audit()
    destination = ROOT / 'reviews/completed_cohort_cpu_audit.json'
    with destination.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'path': str(destination)}))

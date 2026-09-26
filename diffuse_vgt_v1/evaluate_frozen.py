"""One-shot diffuse final scoring, never training or post-test candidate selection.

CLI: PYTHONPATH=.:pydeps:../pydeps:../code:../analysis python evaluate_frozen.py
The fixed results directory is created exclusively. Failed/partial scoring is
preserved and cannot be silently retried. Approval is externally authored; this
entry never creates it. No real rows, GT, or historical errors are read before
the complete-cohort independent gate. Predictions/metrics use parent functions.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORDER = ('O', 'A', 'B', 'C')
ARMS = ORDER[1:]
MODELS = ('baseline', *ORDER)
BASELINE_SHA = '94d1cbf18b550156d46b6a201f87cc5d05619ab7e73c19983d7785bd382d39bb'
O_SHA = '7670d875b96331df7e3c937748aee40d006e8147710ae77ad4962eabb7291351'
REFERENCE_SHA = '08f27579e27e308b90074beff98747aa24307cc6bae1fe2b990b486c353013be'
SUFFIXES = ('_summary.json', '_per_image.csv', '_patches.npz')
CAMERA = 'nikon_512'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def dependencies():
    # Parent round2 imports evaluate_frozen by name. Put its directory first,
    # and verify the resolved object, to avoid this identically named CLI.
    sys.path.insert(0, str(ROOT.parent / 'analysis'))
    sys.path.insert(0, str(ROOT.parent / 'code'))
    import train_arms as runner
    dg, _ = runner.runtime()
    import compare_frozen_means as common
    import compare_round3_means as supports
    import evaluate_dg as checks
    require(Path(dg.r2.evaluation.__file__).resolve() == ROOT.parent / 'code/evaluate_frozen.py',
            'Wrong parent evaluator import')
    return runner, dg, common, supports, checks


def implementation_hashes():
    paths = [Path(__file__), ROOT.parent / 'code/evaluate_frozen.py']
    paths += [ROOT.parent / 'analysis' / name for name in
              ('compare_frozen_means.py', 'compare_round3_means.py', 'evaluate_dg.py')]
    return {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}


def review_file(item):
    path = (ROOT / item['path']).resolve(strict=True)
    require(path.is_relative_to((ROOT / 'reviews').resolve()) and sha(path) == item['sha256'],
            'Independent review missing/stale/outside reviews')
    return path


def checked_selection(r, dg):
    """Recompute all finite development scores and matched completed histories."""
    import numpy as np
    frozen = r.check_frozen(check_sources=False)
    runs = {arm: r.checked_run(arm) for arm in ARMS}
    for arm in ARMS:
        r.check_gate('formal', arm)
    for cycle in range(417):
        hashes = [runs[a][1][cycle]['source_render_diag']['matched_sampling_sha256'] for a in ARMS]
        require(len(set(hashes)) == 1 and len(hashes[0]) == 64, 'Unmatched draw/mask history')
    ledgers = []
    try:
        ledgers = [np.load(ROOT / 'runs' / (a + '_seed0') / 'endpoint_counts.npz', allow_pickle=False) for a in ARMS]
        for key in dg.r3.LEDGER_KEYS:
            require(all(np.array_equal(ledgers[0][key], item[key]) for item in ledgers[1:]),
                    'Unmatched endpoint consumption')
    finally:
        for ledger in ledgers:
            ledger.close()
    selection = read(ROOT / 'development_selection.json')
    require(selection['status'] == 'development_selected_real_evaluation_not_run'
            and selection['tie_order'] == list(ORDER) and tuple(selection['scores']) == ORDER
            and tuple(selection['results']) == ORDER, 'Incomplete/reordered selection')
    require(selection['freeze_manifest_sha256'] == sha(r.FROZEN / 'manifest.json'), 'Stale selection')
    parent, _, parent_ckpt = dg.checked_parent()
    require(selection['results']['O'] == parent, 'Changed O development result')
    checkpoints = {'O': parent_ckpt, **{a: runs[a][0] for a in ARMS}}
    for arm in ORDER:
        result = selection['results'][arm]
        require(set(result['means']) == set(dg.r2.MODES)
                and all(math.isfinite(v) and v > 0 for v in result['means'].values()), 'Invalid development means')
        score = dg.r3.dev_score(result['means'], frozen['reference_means'])
        require(score == result['score'] == selection['scores'][arm], 'Changed development score')
        require(result['checkpoint_sha256'] == sha(checkpoints[arm]), 'Changed development checkpoint')
    require(selection['checkpoint_sha256'] == {a: sha(p) for a, p in checkpoints.items()}, 'Changed checkpoint cohort')
    require(selection['checkpoint_sha256']['O'] == O_SHA, 'Wrong O checkpoint')
    require(selection['winner'] == r.select_winner(selection['scores']), 'Posthoc winner change')
    require(math.isfinite(selection['selected_at']) and selection['selected_at'] > 0, 'Missing selection time')
    return selection, checkpoints


def approved_context(r, dg):
    require(__debug__ and not sys.flags.optimize, 'Optimized Python disables parent assertions')
    selection, checkpoints = checked_selection(r, dg)
    baseline = ROOT.parent / 'evidence/baseline_nikon_seed0_ckpt_final.pt'
    require(sha(baseline) == BASELINE_SHA, 'Baseline checkpoint changed')
    checkpoints = {'baseline': baseline, **checkpoints}
    bindings = dict(scope='diffuse_final_evaluation',
        freeze_manifest_sha256=sha(r.FROZEN / 'manifest.json'),
        development_selection_sha256=sha(ROOT / 'development_selection.json'),
        checkpoint_sha256={a: sha(p) for a, p in checkpoints.items()},
        completion_evidence_sha256={a: {name: sha(ROOT / 'runs' / (a + '_seed0') / name)
            for name in ('completion.json', 'history.jsonl', 'manifest.json', 'endpoint_counts.npz')}
            for a in ARMS}, code_hashes=r.code_hashes(), implementation_hashes=implementation_hashes(),
        official_reference_manifest_sha256=REFERENCE_SHA)
    gate_path = r.FROZEN / 'evaluation_approval.json'
    gate = read(gate_path)
    require(gate['status'] == 'ADEQUATE' and all(gate[k] == v for k, v in bindings.items()), 'Stale evaluation approval')
    review_path = review_file(gate['independent_review'])
    review = read(review_path)
    require(review['artifact_verdict'] == 'PASS' and review['contract_verdict'] == 'ADEQUATE'
            and all(review[k] == v for k, v in bindings.items()), 'Evaluation review missing/stale')
    require(math.isfinite(review['reviewed_at']) and review['reviewed_at'] >= selection['selected_at'],
            'Review predates selection')
    review_file(review['report_markdown'])
    return dict(selection=selection, checkpoints=checkpoints, bindings=bindings,
                approval_sha256=sha(gate_path), review_sha256=sha(review_path))


def artifact_hashes(out):
    return {s + '_' + m + suffix: sha(out / (s + '_' + m + suffix))
            for s in ('val', 'test') for m in MODELS for suffix in SUFFIXES}


def checked_live_metadata(r, dg):
    """Call only after approval; bind the actual parent camera_rows inputs.

    Frozen domislovic_v2.camera_rows sets root=data_root/camera and reads
    root/meta.json and root/split.json (lines 151-154). The same DATA and CAMERA
    objects are passed below, and the parent implementation is hash-frozen.
    Identical image IDs alone cannot detect a changed endpoint whitepoint.
    """
    sources = read(r.FROZEN / 'sources.json')
    camera_root = Path(dg.DATA) / CAMERA
    hashes = {}
    for name in ('split', 'meta'):
        digest = sha(camera_root / (name + '.json'))
        require(digest == sources[name + '_sha256'], 'Live camera metadata changed: ' + name)
        hashes[name + '_sha256'] = digest
    return hashes


def paired_deltas(out, ids):
    """Descriptive equal-image means within scene; compositions are not replicates."""
    image_rows, scene_rows = [], []
    for subset in ('val', 'test'):
        data = {}
        for arm in ARMS:
            with (out / f'{subset}_{arm}_per_image.csv').open(newline='') as f:
                data[arm] = list(csv.DictReader(f))
            require([r['id'] for r in data[arm]] == ids[subset], 'Delta identities differ')
        for idx, image_id in enumerate(ids[subset]):
            for control in ('A', 'C'):
                b, c = data['B'][idx], data[control][idx]
                require(all(b[k] == c[k] for k in ('n_lights', 'valid_patches', 'valid_pixels')), 'Delta support differs')
                row = dict(split=subset, contrast='B-' + control, id=image_id,
                           scene=image_id.rsplit('_', 1)[0], n_lights=int(b['n_lights']))
                for metric in ('patch_mean', 'pixel_mean'):
                    row[metric + '_delta'] = float(b[metric]) - float(c[metric])
                    require(math.isfinite(row[metric + '_delta']), 'Nonfinite paired delta')
                image_rows.append(row)
        for contrast in ('B-A', 'B-C'):
            for scene in sorted({x.rsplit('_', 1)[0] for x in ids[subset]}):
                rows = [r for r in image_rows if r['split'] == subset and r['contrast'] == contrast and r['scene'] == scene]
                for group in ('all', 'single', 'multi'):
                    use = [r for r in rows if group == 'all' or (r['n_lights'] == 1) == (group == 'single')]
                    if not use:
                        continue
                    row = dict(split=subset, contrast=contrast, scene=scene, group=group, compositions=len(use))
                    for metric in ('patch_mean_delta', 'pixel_mean_delta'):
                        row[metric] = sum(r[metric] for r in use) / len(use)
                    scene_rows.append(row)
    for name, rows in (('per_image_deltas.csv', image_rows), ('per_scene_deltas.csv', scene_rows)):
        with (out / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return {name: sha(out / name) for name in ('per_image_deltas.csv', 'per_scene_deltas.csv')}


def aggregate_results(out, context, report, common, supports, checks):
    import numpy as np
    ids = {s: report[s + '_image_ids'] for s in ('val', 'test')}
    metrics, counts = {}, {}
    for model in MODELS:
        metrics[model], counts[model] = {}, {}
        for subset in ids:
            mask, support = supports.support_signature(out, subset, model)
            bmask, bsupport = supports.support_signature(out, subset, 'baseline')
            require(np.array_equal(mask, bmask) and support == bsupport, 'Evaluation support differs')
            summary = read(out / f'{subset}_{model}_summary.json')
            require(summary == report['splits'][subset][model] and summary['model'] == model
                    and summary['split'] == subset and summary['n_images'] == len(ids[subset])
                    and summary['checkpoint_sha256'] == context['bindings']['checkpoint_sha256'][model], 'Summary identity changed')
            rows = checks.checked_pixel_summary(out, subset, model, summary)
            require(all(r['model'] == model and r['split'] == subset for r in rows), 'CSV identity changed')
            values, sizes = common.aggregate(out, subset, model, ids[subset])
            metrics[model].update(values); counts[model].update(sizes)
        require(counts[model] == counts['baseline'], 'Aggregation support differs')
    comparisons = {m: common.compare(metrics[m], metrics['baseline']) for m in ORDER}
    winner = context['selection']['winner']
    passed = winner in ARMS and comparisons[winner]['all_means_strictly_lower']
    return dict(status='complete', outcome='SEED0_GATE_MET' if passed else 'FINAL_TARGET_NOT_MET',
        purpose='acceptance only; never candidate selection', metric_cells=18, frozen_winner=winner,
        winner_passes_seed0_gate=passed, multi_seed_goal_complete=False, metrics=metrics,
        comparisons=comparisons, support=counts['baseline'],
        mechanism_delta_degrees={f'B-{control}': {k: metrics['B'][k] - metrics[control][k] for k in metrics['B']}
                                 for control in ('A', 'C')},
        paired_artifacts_sha256=paired_deltas(out, ids),
        note='Prior test viewing disclosed in USER_TASK. This single seed is not significance; no new training authorized. Scene means group compositions, not independent observations.')


def execute(r, dg, common, supports, checks):
    context = approved_context(r, dg)  # No real data access above this line.
    started = time.time()
    require(started > context['selection']['selected_at'], 'Evaluation predates selection')
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    write_new(out / 'evaluation_start.json', dict(status='running', created=started,
              argv=sys.argv, bindings=context['bindings'], approval_sha256=context['approval_sha256'],
              review_sha256=context['review_sha256']))
    try:
        ref = dg.R2 / 'final_eval/evaluation_manifest.json'
        require(sha(ref) == REFERENCE_SHA, 'Official reference changed')
        reference = read(ref)
        require(reference['status'] == 'complete', 'Incomplete official reference')
        dg.r3.gpu_preflight()
        metadata_hashes = checked_live_metadata(r, dg)
        rows_all = dg.base.camera_rows(dg.DATA, CAMERA)
        report = dict(status='running', created=started, selection=context['selection'],
                      bindings=context['bindings'], live_metadata_sha256=metadata_hashes, splits={})
        with dg.heartbeat(out, stage='diffuse_final_evaluation') as state:
            for subset, count in (('val', 394), ('test', 204)):
                rows = [row for row in rows_all if row['split'] == subset]
                ids = [row['id'] for row in rows]
                require(ids == reference[subset + '_image_ids'] and len(ids) == len(set(ids)) == count,
                        'Official evaluation identities changed')
                report[subset + '_image_ids'] = ids
                report['splits'][subset] = {}
                loaded = dg.base.load_split(rows, 'cuda', 16, subset)
                for model in MODELS:
                    state.update(subset=subset, model=model)
                    report['splits'][subset][model] = dg.r2.evaluation.evaluate_one(
                        model, context['checkpoints'][model], rows, loaded, 'cuda', out, subset)
                del loaded
            require(not {x.rsplit('_', 1)[0] for x in report['val_image_ids']} &
                    {x.rsplit('_', 1)[0] for x in report['test_image_ids']}, 'Val/test scenes overlap')
            require(approved_context(r, dg) == context, 'Authorities changed during scoring')
            require(checked_live_metadata(r, dg) == metadata_hashes, 'Live camera metadata changed during scoring')
            evidence = artifact_hashes(out)
            summary = aggregate_results(out, context, report, common, supports, checks)
            require(artifact_hashes(out) == evidence, 'Evidence changed during aggregation')
            report.update(status='complete', completed=time.time(), evidence_sha256=evidence,
                          approval_sha256=context['approval_sha256'], review_sha256=context['review_sha256'])
            write_new(out / 'evaluation_manifest.json', report)
            summary.update(evaluation_manifest_sha256=sha(out / 'evaluation_manifest.json'), **context['bindings'])
            write_new(out / 'summary.json', summary)
            state.update(status='complete')
        return summary
    except Exception as exc:
        write_new(out / 'failure.json', dict(status='FAILED', error=repr(exc), created=started,
                                            failed_at=time.time(), bindings=context['bindings']))
        raise


def main():
    require(len(sys.argv) == 1, 'No mutable CLI parameters; use frozen protocol')
    r, dg, common, supports, checks = dependencies()
    with dg.r3.exclusive_lock():
        execute(r, dg, common, supports, checks)


if __name__ == '__main__':
    main()

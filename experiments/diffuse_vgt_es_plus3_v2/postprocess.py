"""Finite seven-arm final scoring and evidence report; never train or select on test.

Commands: bindings (read-only JSON), evaluate (external approval required), report.
No existing result directory is resumed or overwritten after partial scoring.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
CELLS = tuple(f'{s}/{g}/{m}' for s in ('val', 'test')
              for g in ('all', 'single', 'multi')
              for m in ('patch_pooled', 'patch_image_balanced', 'pixel_image_balanced'))
CONTRASTS = {
    'ES10-ES00': {'ES10': 1, 'ES00': -1},
    'ES11-ES01': {'ES11': 1, 'ES01': -1},
    'ES01-ES00': {'ES01': 1, 'ES00': -1},
    'ES11-ES10': {'ES11': 1, 'ES10': -1},
    'ES11-ES10-ES01+ES00': {'ES11': 1, 'ES10': -1, 'ES01': -1, 'ES00': 1},
    'ES11-KEEP_OLD_R': {'ES11': 1, 'KEEP_OLD_R': -1},
    'ES00-SHAM_OLD_0': {'ES00': 1, 'SHAM_OLD_0': -1},
    'NO_NEW_S-ES00': {'NO_NEW_S': 1, 'ES00': -1},
    'NO_NEW_S-ES01': {'NO_NEW_S': 1, 'ES01': -1},
}


def dependencies():
    import queue_train as q
    spec = importlib.util.spec_from_file_location('parent_final_scoring', q.PARENT / 'evaluate_frozen.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    sys.path.insert(0, str(q.PARENT.parent / 'analysis'))
    sys.path.insert(0, str(q.PARENT.parent / 'code'))
    dg, _ = q.runtime()
    import compare_frozen_means as common
    import compare_round3_means as supports
    import evaluate_dg as checks
    q.require(Path(dg.r2.evaluation.__file__).resolve() == q.PARENT.parent / 'code/evaluate_frozen.py',
              'Wrong official evaluator')
    return q, old, dg, common, supports, checks


def checked_context(q, old, dg):
    """No real camera rows, real GT or real result payloads are accessed here."""
    import numpy as np
    q.require(__debug__ and not sys.flags.optimize, 'Optimized Python prohibited')
    frozen = q.check_frozen(False)
    queue = q.state()
    q.require(queue['queue'] == list(q.ARMS) and queue['max_successful_updates'] == 487200,
              'Changed queue budget/order')
    runs = {}
    for arm in q.ARMS:
        q.require(queue['arms'][arm]['status'] == 'COMPLETE' and
                  queue['arms'][arm]['successful_updates'] == 69600, 'Incomplete queue arm')
        q.gate('formal', arm)
        runs[arm] = q.engine.checked_run(arm)
    for cycle in range(417):
        rows = [runs[a][1][cycle] for a in q.ARMS]
        hashes = {r['source_render_diag']['matched_sampling_sha256'] for r in rows}
        q.require(len(hashes) == 1 and len(next(iter(hashes))) == 64, 'Unmatched sampling/mask')
        consumption = [[u['valid_patches'] for u in r['updates']] for r in rows]
        q.require(all(v == consumption[0] for v in consumption), 'Unmatched actual patch consumption')
    ledgers = []
    try:
        ledgers = [np.load(ROOT / 'runs' / f'{a}_seed0' / 'endpoint_counts.npz', allow_pickle=False)
                   for a in q.ARMS]
        for key in dg.r3.LEDGER_KEYS:
            q.require(all(np.array_equal(ledgers[0][key], x[key]) for x in ledgers[1:]),
                      'Unmatched endpoint consumption')
    finally:
        for ledger in ledgers:
            ledger.close()
    path = ROOT / 'selection_before_real_eval.json'
    selection = q.read(path)
    q.require(selection == q.read(ROOT / 'development_selection.json'), 'Selection copies differ')
    q.require(selection['status'] == 'development_selected_real_evaluation_not_run' and
              selection['tie_order'] == list(q.TIE) and set(selection['scores']) == set(q.TIE) and
              set(selection['results']) == set(q.TIE), 'Incomplete development cohort or changed tie order')
    q.require(selection['freeze_manifest_sha256'] == q.sha(q.FROZEN / 'manifest.json'), 'Stale selection')
    parent, _, parent_ckpt = dg.checked_parent()
    q.require(selection['results']['O'] == parent, 'Changed inherited O development score')
    checkpoints = {a: parent_ckpt if a == 'O' else runs[a][0] for a in q.TIE}
    for arm in q.TIE:
        result = selection['results'][arm]
        q.require(set(result['means']) == set(dg.r2.MODES) and
                  all(math.isfinite(x) and x > 0 for x in result['means'].values()), 'Invalid dev means')
        score = dg.r3.dev_score(result['means'], frozen['reference_means'])
        q.require(score == result['score'] == selection['scores'][arm], 'Changed dev score')
        q.require(result['checkpoint_sha256'] == q.sha(checkpoints[arm]), 'Changed dev checkpoint')
    q.require(selection['winner'] == min(q.TIE, key=selection['scores'].get), 'Post-test winner change')
    q.require(math.isfinite(selection['selected_at']) and selection['selected_at'] > 0, 'Invalid selection time')
    q.require(selection['checkpoint_sha256'] == {a: q.sha(p) for a, p in checkpoints.items()},
              'Changed selected checkpoints')
    q.require(q.sha(parent_ckpt) == old.O_SHA, 'Wrong inherited O checkpoint')
    baseline = q.PARENT.parent / 'evidence/baseline_nikon_seed0_ckpt_final.pt'
    q.require(q.sha(baseline) == old.BASELINE_SHA, 'Wrong frozen baseline')
    checkpoints = {'baseline': baseline, **checkpoints}
    implementation = [Path(__file__), q.PARENT / 'evaluate_frozen.py',
                      q.PARENT.parent / 'code/evaluate_frozen.py']
    implementation += [q.PARENT.parent / 'analysis' / n for n in
                       ('compare_frozen_means.py', 'compare_round3_means.py', 'evaluate_dg.py')]
    bindings = dict(scope='es_plus3_final_evaluation',
        freeze_manifest_sha256=q.sha(q.FROZEN / 'manifest.json'),
        selection_before_real_eval_sha256=q.sha(path),
        checkpoint_sha256={a: q.sha(p) for a, p in checkpoints.items()},
        completion_evidence_sha256={a: {n: q.sha(ROOT / 'runs' / f'{a}_seed0' / n)
            for n in ('completion.json', 'history.jsonl', 'manifest.json', 'endpoint_counts.npz')}
            for a in q.ARMS}, code_hashes=q.code_hashes(),
        implementation_hashes={str(p): q.sha(p) for p in implementation},
        official_reference_manifest_sha256=old.REFERENCE_SHA,
        diagnostic_sha256={n: q.sha(ROOT/n) for n in
                           ('real24_qc_diagnostics_v2/summary.json', 'input_signal.json')})
    return dict(selection=selection, checkpoints=checkpoints, bindings=bindings)


def approved_context(q, old, dg):
    context = checked_context(q, old, dg)
    gate_path = q.FROZEN / 'evaluation_approval.json'
    gate = q.read(gate_path)
    b = context['bindings']
    q.require(gate['status'] == 'ADEQUATE' and all(gate[k] == v for k, v in b.items()), 'Stale evaluation gate')
    def review_file(item):
        p = (ROOT / item['path']).resolve(strict=True)
        q.require(p.is_relative_to((ROOT / 'reviews').resolve()) and q.sha(p) == item['sha256'],
                  'Missing/stale independent review')
        return p
    review_path = review_file(gate['independent_review'])
    review = q.read(review_path)
    q.require(review['artifact_verdict'] == 'PASS' and review['contract_verdict'] == 'ADEQUATE' and
              all(review[k] == v for k, v in b.items()), 'Evaluation review inadequate')
    q.require(math.isfinite(review['reviewed_at']) and review['reviewed_at'] >=
              context['selection']['selected_at'], 'Review predates selection')
    review_file(review['report_markdown'])
    context.update(approval_sha256=q.sha(gate_path), review_sha256=q.sha(review_path))
    return context


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contrasts(metrics):
    return {name: {cell: sum(sign * metrics[arm][cell] for arm, sign in coefficients.items())
                   for cell in CELLS} for name, coefficients in CONTRASTS.items()}


def aggregate(out, context, report, q, common, supports, checks):
    import numpy as np
    metrics, counts, image_rows, scene_rows = {}, {}, [], []
    models = tuple(context['checkpoints'])
    for model in models:
        metrics[model], counts[model] = {}, {}
        for split in ('val', 'test'):
            mask, sig = supports.support_signature(out, split, model)
            bmask, bsig = supports.support_signature(out, split, 'baseline')
            q.require(np.array_equal(mask, bmask) and sig == bsig, 'Real evaluation support differs')
            summary = q.read(out / f'{split}_{model}_summary.json')
            q.require(summary == report['splits'][split][model] and summary['model'] == model and
                      summary['split'] == split and summary['n_images'] == len(report[split+'_image_ids']) and
                      summary['checkpoint_sha256'] == context['bindings']['checkpoint_sha256'][model],
                      'Changed scoring identity')
            rows = checks.checked_pixel_summary(out, split, model, summary)
            q.require([r['id'] for r in rows] == report[split+'_image_ids'] and
                      all(r['model'] == model and r['split'] == split for r in rows), 'CSV identities changed')
            values, sizes = common.aggregate(out, split, model, report[split+'_image_ids'])
            metrics[model].update(values)
            counts[model].update(sizes)
            for row in rows:
                image_rows.append(dict(model=model, split=split, id=row['id'],
                    scene=row['id'].rsplit('_', 1)[0], n_lights=int(row['n_lights']),
                    valid_patches=int(row['valid_patches']), valid_pixels=int(row['valid_pixels']),
                    patch_mean=float(row['patch_mean']), pixel_mean=float(row['pixel_mean'])))
        q.require(counts[model] == counts['baseline'], 'Aggregation support differs')
    for model in models:
        for split in ('val', 'test'):
            rows = [r for r in image_rows if r['model'] == model and r['split'] == split]
            for scene in sorted({r['scene'] for r in rows}):
                for group in ('all', 'single', 'multi'):
                    use = [r for r in rows if r['scene'] == scene and
                           (group == 'all' or (r['n_lights'] == 1) == (group == 'single'))]
                    if use:
                        scene_rows.append(dict(model=model, split=split, scene=scene, group=group,
                            compositions=len(use), patch_image_balanced=sum(r['patch_mean'] for r in use)/len(use),
                            pixel_image_balanced=sum(r['pixel_mean'] for r in use)/len(use)))
    delta = contrasts(metrics)
    write_csv(out / 'per_image.csv', image_rows)
    write_csv(out / 'per_scene.csv', scene_rows)
    write_csv(out / 'contrasts.csv', [dict(contrast=name, cell=cell, delta_degrees=value)
              for name, values in delta.items() for cell, value in values.items()])
    comparisons = {m: common.compare(metrics[m], metrics['baseline']) for m in q.TIE}
    winner = context['selection']['winner']
    passed = winner in q.ARMS and comparisons[winner]['all_means_strictly_lower']
    return dict(status='complete', outcome='SEED0_GATE_MET' if passed else 'FINAL_TARGET_NOT_MET',
        frozen_winner=winner, winner_passes_seed0_gate=passed, multi_seed_goal_complete=False,
        metric_cells=18, metrics=metrics, comparisons=comparisons, support=counts['baseline'],
        mechanism_delta_degrees=delta, bootstrap='NOT_RUN; optional descriptive scene bootstrap omitted',
        paired_artifacts_sha256={n: q.sha(out/n) for n in ('per_image.csv', 'per_scene.csv', 'contrasts.csv')},
        note='One seed, historical test viewed; no test selection or physical accuracy claim.')


def evaluate(q, old, dg, common, supports, checks):
    context = approved_context(q, old, dg)  # All checks above precede real data access.
    started = time.time()
    q.require(started > context['selection']['selected_at'], 'Scoring precedes selection')
    out = ROOT / 'results'
    out.mkdir(exist_ok=False)
    old.write_new(out/'evaluation_start.json', dict(status='running', created=started,
        bindings=context['bindings'], approval_sha256=context['approval_sha256'], review_sha256=context['review_sha256']))
    try:
        ref = dg.R2 / 'final_eval/evaluation_manifest.json'
        q.require(q.sha(ref) == old.REFERENCE_SHA, 'Changed official reference')
        reference = q.read(ref)
        q.require(reference['status'] == 'complete', 'Incomplete official reference')
        metadata = old.checked_live_metadata(q, dg)
        dg.r3.gpu_preflight()
        rows_all = dg.base.camera_rows(dg.DATA, old.CAMERA)
        report = dict(status='running', created=started, selection=context['selection'],
            bindings=context['bindings'], live_metadata_sha256=metadata, splits={})
        for split, count in (('val', 394), ('test', 204)):
            rows = [r for r in rows_all if r['split'] == split]
            ids = [r['id'] for r in rows]
            q.require(ids == reference[split+'_image_ids'] and len(ids) == len(set(ids)) == count,
                      'Changed official val/test identities')
            report[split+'_image_ids'] = ids
            report['splits'][split] = {}
            loaded = dg.base.load_split(rows, 'cuda', 16, split)
            for model, checkpoint in context['checkpoints'].items():
                report['splits'][split][model] = dg.r2.evaluation.evaluate_one(
                    model, checkpoint, rows, loaded, 'cuda', out, split)
            del loaded
        q.require(not {x.rsplit('_',1)[0] for x in report['val_image_ids']} &
                  {x.rsplit('_',1)[0] for x in report['test_image_ids']}, 'Val/test scene overlap')
        q.require(approved_context(q, old, dg) == context, 'Authorities changed during scoring')
        q.require(old.checked_live_metadata(q, dg) == metadata, 'Metadata changed during scoring')
        evidence = {f'{s}_{m}{suffix}': q.sha(out/f'{s}_{m}{suffix}')
                    for s in ('val','test') for m in context['checkpoints'] for suffix in old.SUFFIXES}
        summary = aggregate(out, context, report, q, common, supports, checks)
        q.require(all(q.sha(out/n) == h for n,h in evidence.items()), 'Scoring artifacts changed')
        report.update(status='complete', completed=time.time(), evidence_sha256=evidence)
        old.write_new(out/'evaluation_manifest.json', report)
        summary.update(evaluation_manifest_sha256=q.sha(out/'evaluation_manifest.json'), **context['bindings'])
        old.write_new(out/'summary.json', summary)
    except Exception as exc:
        old.write_new(out/'failure.json',dict(status='FAILED', error=repr(exc), failed_at=time.time(),
                                            bindings=context['bindings']))
        raise


def report(q):
    """CPU-only report: read saved evidence, no fresh prediction or GT access."""
    out = ROOT/'results'
    q.require(not (out/'failure.json').exists(), 'Scoring failure retained; no partial report')
    s, manifest = q.read(out/'summary.json'), q.read(out/'evaluation_manifest.json')
    selection = q.read(ROOT/'selection_before_real_eval.json')
    q.require(s['status'] == manifest['status'] == 'complete' and s['metric_cells'] == 18 and
              s['evaluation_manifest_sha256'] == q.sha(out/'evaluation_manifest.json') and
              s['selection_before_real_eval_sha256'] == q.sha(ROOT/'selection_before_real_eval.json') and
              s['freeze_manifest_sha256'] == q.sha(q.FROZEN/'manifest.json') and
              selection == manifest['selection'] and s['frozen_winner'] == selection['winner'] ==
              min(q.TIE,key=selection['scores'].get), 'Stale report authorities')
    for name,h in {**manifest['evidence_sha256'], **s['paired_artifacts_sha256']}.items():
        q.require(q.sha(out/name) == h, 'Changed report evidence')
    for name,h in s['diagnostic_sha256'].items():
        q.require(q.sha(ROOT/name) == h, 'Changed diagnostic evidence')
    q.require(set(s['metrics']) == {'baseline',*q.TIE} and
              all(set(v) == set(CELLS) and all(math.isfinite(x) and x >= 0 for x in v.values())
                  for v in s['metrics'].values()), 'Incomplete metric cohort')
    q.require(s['mechanism_delta_degrees'] == contrasts(s['metrics']), 'Changed contrast values')
    winner = s['frozen_winner']
    comparisons = {a: sum(s['metrics'][a][c] < s['metrics']['baseline'][c] for c in CELLS) for a in q.TIE}
    passed = winner in q.ARMS and comparisons[winner] == 18
    q.require(passed == s['winner_passes_seed0_gate'] and
              s['outcome'] == ('SEED0_GATE_MET' if passed else 'FINAL_TARGET_NOT_MET'), 'Invalid outcome')
    lines = ['# ES + three controls: seed0 final report', '', 'agent: codex',
        f"updated: {time.strftime('%Y-%m-%d')}\n",
        f"Frozen development winner: {winner}. Outcome: {s['outcome']}. Selection timestamp: {selection['selected_at']}.",
        'Acceptance requires the frozen new winner to beat baseline strictly in all 18 cells. Multi-seed goal is incomplete.',
        'Angles are degrees (lower is better), measured with evaluation GT unavailable in real inference. Historical test was viewed. No test tuning occurred in this scoring entry.',
        'O/baseline retain earlier data and prior/mask conditions; seven arms share the new mask and exposure. Differences versus O do not isolate a single mechanism.',
        'Patch pooled weights patches; patch image-balanced and pixel image-balanced average image means. Scene rows average compositions; compositions are not independent scenes.',
        'External Intrinsic/DSINE estimates and numerical I/E=J consistency do not establish physical accuracy. TIFF/black-level provenance remains the inherited engineering assumption.',
        'Optional bootstrap was omitted. One training seed cannot establish significance or seed uncertainty.', '',
        '| Candidate | Frozen dev score | Cells below baseline /18 |', '| --- | --- | --- |']
    lines += [f'| {a} | {selection["scores"][a]:.9g} | {comparisons[a]} |' for a in q.TIE]
    modes = tuple(selection['results']['O']['means'])
    lines += ['', '| Development component | '+' | '.join(q.TIE)+' |',
              '| --- | '+' | '.join('---' for _ in q.TIE)+' |']
    lines += ['| '+mode+' | '+' | '.join(f'{selection["results"][a]["means"][mode]:.9g}'
               for a in q.TIE)+' |' for mode in modes]
    models = ('baseline',*q.TIE)
    lines += ['', '| Cell | '+' | '.join(models)+' |', '| --- | '+' | '.join('---' for _ in models)+' |']
    lines += ['| '+c+' | '+' | '.join(f'{s["metrics"][a][c]:.9g}' for a in models)+' |' for c in CELLS]
    lines += ['', 'Predeclared contrasts: negative means the first expression has lower error. Interaction has its declared four-term sign.',
              '| Cell | '+' | '.join(CONTRASTS)+' |', '| --- | '+' | '.join('---' for _ in CONTRASTS)+' |']
    lines += ['| '+c+' | '+' | '.join(f'{s["mechanism_delta_degrees"][a][c]:.9g}' for a in CONTRASTS)+' |' for c in CELLS]
    lines += ['', '| Contrast | Lower error cells | Higher error cells | Equal cells |', '| --- | --- | --- | --- |']
    for name, values in s['mechanism_delta_degrees'].items():
        lines.append(f'| {name} | {sum(x < 0 for x in values.values())} | {sum(x > 0 for x in values.values())} | {sum(x == 0 for x in values.values())} |')
    lines += ['', 'Each contrast is a descriptive observation, including unfavorable and mixed directions; none alone validates a physical prior.',
        'Actual support: `'+json.dumps(s['support'],sort_keys=True)+'`.',
        'Formal successful updates: 7 × 69,600 = 487,200. Calibration is separate (at most 14 updates).',
        'Per-image, per-scene and all contrast cells: results/per_image.csv, results/per_scene.csv, results/contrasts.csv.',
        'Frozen checkpoint SHA256: `'+json.dumps(s['checkpoint_sha256'],sort_keys=True)+'`.',
        'Implementation SHA256: `'+json.dumps(s['implementation_hashes'],sort_keys=True)+'`.',
        'This private report retains complete evidence bindings; public synchronization requires separate sanitization and independent final validation.']
    diagnostics = q.read(ROOT/'real24_qc_diagnostics_v2/summary.json')
    q.require(diagnostics['status'] == 'PASS' and diagnostics['count'] == len(diagnostics['scenes']) == 24,
              'Incomplete fixed24 diagnostics')
    lines += ['', 'Intervention strength: fixed24 full-view formula previews (not the complete training crop/visit stream).',
              '| Descriptor | Mean across fixed24 | Minimum | Maximum |', '| --- | --- | --- | --- |']
    for field in ('E_rotation_mean_deg','E_rotation_patch_mean_deg','S_rotation_log_rms',
                  'S_rotation_patch_log_rms','s0_rotation_log_rms','s0_rotation_patch_log_rms'):
        values = [r[field] for r in diagnostics['scenes']]
        q.require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid intervention descriptor')
        lines.append(f'| {field} | {sum(values)/len(values):.9g} | {min(values):.9g} | {max(values):.9g} |')
    lines += ['', 'Normalized global/local input differences: input_signal.json (full/tail previews, first16 mixed views; not all training visits).',
              'Diagnostic evidence SHA256: `'+json.dumps(s['diagnostic_sha256'],sort_keys=True)+'`.']
    with (ROOT/'FINAL_REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('bindings','evaluate','report'))
    args = p.parse_args()
    if args.mode == 'report':
        import queue_train as q
        report(q)
        return
    q, old, dg, common, supports, checks = dependencies()
    with dg.r3.exclusive_lock():
        if args.mode == 'bindings':
            print(json.dumps(checked_context(q,old,dg)['bindings'], indent=2, allow_nan=False))
        else:
            evaluate(q,old,dg,common,supports,checks)
            report(q)


if __name__ == '__main__':
    main()

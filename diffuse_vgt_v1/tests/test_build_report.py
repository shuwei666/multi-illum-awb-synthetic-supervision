"""Synthetic, temporary report fixtures only; never reads experiment results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('report', Path(__file__).parents[1] / 'build_report.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    write(root, 'frozen/manifest.json', {'steps': 69600})
    digest = report.sha(root / 'frozen/manifest.json')
    for arm in 'ABC':
        write(root, f'runs/{arm}_seed0/completion.json', dict(status='training_complete', arm=arm, steps=69600, cycles=2, freeze_manifest_sha256=digest, seconds=200, peak_gpu_bytes=300))
        records = [dict(online_seconds_per_update=1, updates=[{}] * 60000), dict(online_seconds_per_update=3, updates=[{}] * 9600)]
        for row in records:
            n = len(row['updates'])
            row.update(consumed_views=n * 8, valid_patches=n * 500,
                       source_render_diag=dict(consumed_mixed_views=n * 4, consumed_mixed_valid_patches=n * 250))
        (root / f'runs/{arm}_seed0/history.jsonl').write_text('\n'.join(json.dumps(x) for x in records))
        completion = report.read(root / f'runs/{arm}_seed0/completion.json')
        completion['outputs'] = {'history.jsonl': report.sha(root / f'runs/{arm}_seed0/history.jsonl')}
        write(root, f'runs/{arm}_seed0/completion.json', completion)
        write(root, f'calibration/attempt_001/{arm}/completion.json', dict(status='calibration_complete', steps=2, seconds=4, peak_gpu_bytes=5))
    selection = dict(status='development_selected_real_evaluation_not_run', freeze_manifest_sha256=digest, tie_order=list(report.MODELS[1:]), winner='B', scores={'O': 4, 'A': 3, 'B': 1, 'C': 2}, results={a: {'means': {m: i + 1 for m in report.MODES}} for i, a in enumerate(report.MODELS[1:])})
    write(root, 'development_selection.json', selection)
    write(root, 'results/evaluation_manifest.json', {'status': 'complete'})
    metrics = {a: {k: float(v) for k in report.CELLS} for a, v in zip(report.MODELS, (5, 4, 3, 1, 2))}
    summary = dict(status='complete', metric_cells=18, development_selection_sha256=report.sha(root / 'development_selection.json'), freeze_manifest_sha256=digest, frozen_winner='B', evaluation_manifest_sha256=report.sha(root / 'results/evaluation_manifest.json'), completion_evidence_sha256={a: {'completion.json': report.sha(root / f'runs/{a}_seed0/completion.json')} for a in 'ABC'}, metrics=metrics, winner_passes_seed0_gate=True, outcome='SEED0_GATE_MET', multi_seed_goal_complete=False, mechanism_delta_degrees={f'B-{a}': {k: metrics['B'][k] - metrics[a][k] for k in report.CELLS} for a in 'AC'})
    for arm in 'ABC':
        summary['completion_evidence_sha256'][arm]['history.jsonl'] = report.sha(root / f'runs/{arm}_seed0/history.jsonl')
    summary['paired_artifacts_sha256'] = {}
    for name in ('per_image_deltas.csv', 'per_scene_deltas.csv'):
        (root / 'results' / name).write_text('mock_only\n1\n')
        summary['paired_artifacts_sha256'][name] = report.sha(root / 'results' / name)
    write(root, 'results/summary.json', summary)
    write(root, 'cache_manifest.json', dict(seconds=8, files={'mock': {}}, bytes=9, unusable_source_fraction=0))
    write(root, 'cache_progress.json', dict(peak_gpu_bytes=123456))
    write(root, 'preflight_001.json', dict(seconds=2, peak_allocated_bytes=3, scope='mock'))
    write(root, 'priors_manifest.json', dict(licenses='mock', external_training_sources='mock', nikon_overlap='UNKNOWN'))
    for path in ('USER_TASK.txt', 'REVISION_1.md', 'run_manifest.json', 'build_report.py'):
        write(root, path, 'MOCK ONLY')
    return summary


class ReportTest(unittest.TestCase):
    def test_mock_render_and_cli_writes_only_temp(self):
        with tempfile.TemporaryDirectory(prefix='diffuse-report-mock-') as directory:
            root = Path(directory)
            fixture(root)
            md, page = report.render(root)
            self.assertIn('待独立审查', md)
            self.assertIn('18格中的18格', md)
            self.assertIn('123456', md)
            self.assertIn(str(69600 * 8), md)
            self.assertIn(str(69600 * 500), md)
            self.assertIn(str(69600 * 4), md)
            self.assertIn(str(69600 * 250), md)
            self.assertIn('不是独立照片', md)
            self.assertIn(str((60000 + 3 * 9600) / 69600), md)
            self.assertEqual(page.count('<section class="sec"'), 8)
            self.assertEqual(page.count('</section>'), 8)
            self.assertIn('max-width:1100px', page)
            self.assertNotIn('<script src=', page)
            for cell in report.CELLS:
                self.assertIn(cell, md)
            with patch.object(report, 'ROOT', root), patch('sys.argv', ['build_report.py']):
                report.main()
                self.assertEqual((root / 'FINAL_REPORT.md').read_text(), md)
                self.assertEqual((root / 'FINAL_REPORT.html').read_text(), page)
                with self.assertRaisesRegex(RuntimeError, 'Existing report'):
                    report.main()

    def test_missing_consumption_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='diffuse-report-mock-') as directory:
            root = Path(directory)
            fixture(root)
            path = root / 'runs/A_seed0/history.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            del records[0]['source_render_diag']['consumed_mixed_valid_patches']
            path.write_text('\n'.join(json.dumps(row) for row in records))
            digest = report.sha(path)
            completion = report.read(root / 'runs/A_seed0/completion.json')
            completion['outputs']['history.jsonl'] = digest
            write(root, 'runs/A_seed0/completion.json', completion)
            summary = report.read(root / 'results/summary.json')
            summary['completion_evidence_sha256']['A'].update({'history.jsonl': digest, 'completion.json': report.sha(root / 'runs/A_seed0/completion.json')})
            write(root, 'results/summary.json', summary)
            with self.assertRaises(KeyError):
                report.render(root)
            self.assertFalse((root / 'FINAL_REPORT.md').exists())

    def test_stale_history_and_paired_artifacts_rejected(self):
        for target in ('runs/A_seed0/history.jsonl', 'results/per_image_deltas.csv', 'results/per_scene_deltas.csv'):
            for missing in (False, True):
                with self.subTest(target=target, missing=missing), tempfile.TemporaryDirectory(prefix='diffuse-report-mock-') as directory:
                    root = Path(directory)
                    fixture(root)
                    path = root / target
                    if missing:
                        path.unlink()
                    else:
                        path.write_text(path.read_text() + '\n')
                    with self.assertRaises((RuntimeError, FileNotFoundError)):
                        report.load(root)

    def test_private_review_and_fields_rejected_without_rewriting(self):
        for text in ('/Users/mockperson/private/project', 'mockuser@private.example', 'ssh://myhost', '10.1.2.3', '/Volumes/private/data'):
            for target in ('review', 'prior', 'preflight'):
                with self.subTest(text=text, target=target), tempfile.TemporaryDirectory(prefix='diffuse-report-mock-') as directory:
                    root = Path(directory)
                    fixture(root)
                    if target == 'review':
                        path = root / 'reviews/final_metrics_review.md'
                        path.parent.mkdir()
                        path.write_text('Verdict: FAIL\n' + text)
                    else:
                        path = root / ('priors_manifest.json' if target == 'prior' else 'preflight_001.json')
                        data = report.read(path)
                        data['nikon_overlap' if target == 'prior' else 'scope'] = text
                        write(root, str(path.relative_to(root)), data)
                    before = path.read_bytes()
                    with self.assertRaisesRegex(RuntimeError, 'Private content rejected'):
                        report.render(root)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertFalse((root / 'FINAL_REPORT.md').exists())

    def test_completion_history_binding_required(self):
        with tempfile.TemporaryDirectory(prefix='diffuse-report-mock-') as directory:
            root = Path(directory)
            summary = fixture(root)
            completion = report.read(root / 'runs/A_seed0/completion.json')
            completion['outputs']['history.jsonl'] = 'stale'
            write(root, 'runs/A_seed0/completion.json', completion)
            summary['completion_evidence_sha256']['A']['completion.json'] = report.sha(root / 'runs/A_seed0/completion.json')
            write(root, 'results/summary.json', summary)
            with self.assertRaisesRegex(RuntimeError, 'Changed history binding'):
                report.load(root)

    def test_incomplete_and_stale_evidence_rejected(self):
        with tempfile.TemporaryDirectory(prefix='diffuse-report-mock-') as directory:
            root = Path(directory)
            summary = fixture(root)
            summary['status'] = 'running'
            write(root, 'results/summary.json', summary)
            with self.assertRaisesRegex(RuntimeError, 'Evaluation incomplete'):
                report.render(root)
            summary['status'] = 'complete'
            summary['development_selection_sha256'] = 'stale'
            write(root, 'results/summary.json', summary)
            with self.assertRaisesRegex(RuntimeError, 'Stale evaluation binding'):
                report.load(root)
            self.assertFalse((root / 'FINAL_REPORT.md').exists())


if __name__ == '__main__':
    unittest.main()

"""CPU/mock tests only. No real Nikon image, GT or checkpoint is opened."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'evaluate_frozen.py'
spec = importlib.util.spec_from_file_location('diffuse_evaluation_candidate', PATH)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'diffuse'
        self.root.mkdir()
        self.root_patch = patch.object(e, 'ROOT', self.root)
        self.root_patch.start()
        self.r = NS(FROZEN=self.root / 'frozen', code_hashes=lambda: {'runner': 'mock'})
        save(self.r.FROZEN / 'manifest.json', {'mock': 'freeze'})
        save(self.root / 'development_selection.json', {'mock': 'selection'})
        baseline = self.root.parent / 'evidence/baseline_nikon_seed0_ckpt_final.pt'
        save(baseline, {'mock': 'baseline'})
        self.baseline_patch = patch.object(e, 'BASELINE_SHA', e.sha(baseline))
        self.baseline_patch.start()
        self.checkpoints = {}
        for a in e.ORDER:
            path = self.root / (a + '.pt')
            save(path, {'mock': a})
            self.checkpoints[a] = path
        for a in e.ARMS:
            for name in ('completion.json', 'history.jsonl', 'manifest.json', 'endpoint_counts.npz'):
                save(self.root / 'runs' / (a + '_seed0') / name, {'mock': a + name})
        self.selection = {'selected_at': 1.0, 'winner': 'O'}
        self.selection_patch = patch.object(e, 'checked_selection', return_value=(self.selection, self.checkpoints))
        self.selection_patch.start()
        self.impl_patch = patch.object(e, 'implementation_hashes', return_value={'candidate': 'mock'})
        self.impl_patch.start()
        self.bindings = dict(scope='diffuse_final_evaluation',
            freeze_manifest_sha256=e.sha(self.r.FROZEN / 'manifest.json'),
            development_selection_sha256=e.sha(self.root / 'development_selection.json'),
            checkpoint_sha256={a: e.sha(p) for a, p in {'baseline': baseline, **self.checkpoints}.items()},
            completion_evidence_sha256={a: {name: e.sha(self.root / 'runs' / (a + '_seed0') / name)
                for name in ('completion.json', 'history.jsonl', 'manifest.json', 'endpoint_counts.npz')} for a in e.ARMS},
            code_hashes={'runner': 'mock'}, implementation_hashes={'candidate': 'mock'},
            official_reference_manifest_sha256=e.REFERENCE_SHA)
        prose = self.root / 'reviews/mock.md'
        save(prose, {'mock': 'independent review'})
        self.review_path = self.root / 'reviews/mock.json'
        self.review = dict(artifact_verdict='PASS', contract_verdict='ADEQUATE', reviewed_at=2.0,
            report_markdown={'path': 'reviews/mock.md', 'sha256': e.sha(prose)}, **self.bindings)
        self.install_gate()

    def install_gate(self):
        save(self.review_path, self.review)
        save(self.r.FROZEN / 'evaluation_approval.json', dict(status='ADEQUATE',
            independent_review={'path': 'reviews/mock.json', 'sha256': e.sha(self.review_path)}, **self.bindings))

    def tearDown(self):
        for p in (self.impl_patch, self.selection_patch, self.baseline_patch, self.root_patch):
            p.stop()
        self.tmp.cleanup()

    def test_valid_mock_gate(self):
        context = e.approved_context(self.r, NS())
        self.assertEqual(tuple(context['checkpoints']), e.MODELS)
        self.assertFalse((self.root / 'results').exists())

    def test_changed_completion_or_code_rejected(self):
        save(self.root / 'runs/A_seed0/history.jsonl', {'changed': True})
        with self.assertRaisesRegex(RuntimeError, 'approval'):
            e.approved_context(self.r, NS())

    def test_stale_review_and_fail_rejected(self):
        self.review['reviewed_at'] = .5
        self.install_gate()
        with self.assertRaisesRegex(RuntimeError, 'predates'):
            e.approved_context(self.r, NS())
        self.review['reviewed_at'] = 2
        self.review['artifact_verdict'] = 'FAIL'
        self.install_gate()
        with self.assertRaisesRegex(RuntimeError, 'review'):
            e.approved_context(self.r, NS())

    def test_unapproved_execution_never_loads_data_or_creates_results(self):
        (self.r.FROZEN / 'evaluation_approval.json').unlink()
        dg = NS(base=NS(camera_rows=Mock(side_effect=AssertionError('real data touched'))))
        with self.assertRaises(FileNotFoundError):
            e.execute(self.r, dg, None, None, None)
        dg.base.camera_rows.assert_not_called()
        self.assertFalse((self.root / 'results').exists())

    def test_existing_output_never_reopened(self):
        (self.root / 'results').mkdir()
        with self.assertRaises(FileExistsError):
            e.execute(self.r, NS(), None, None, None)

    def test_review_path_escape_rejected(self):
        outside = self.root.parent / 'outside.json'
        save(outside, {})
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            e.review_file({'path': '../outside.json', 'sha256': e.sha(outside)})


class SelectionTests(unittest.TestCase):
    def test_real_selection_validation_on_synthetic_cohort(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, 'ROOT', Path(temp)):
            root = Path(temp)
            frozen = root / 'frozen'
            save(frozen / 'manifest.json', {})
            ckpts, histories, results = {}, {}, {}
            for arm in e.ORDER:
                ckpts[arm] = root / (arm + '.pt')
                save(ckpts[arm], {'mock': arm})
                results[arm] = dict(means={'single': 1.0}, score=1.0, checkpoint_sha256=e.sha(ckpts[arm]))
            for arm in e.ARMS:
                histories[arm] = [{'source_render_diag': {'matched_sampling_sha256': 'a' * 64}} for _ in range(417)]
                directory = root / 'runs' / (arm + '_seed0')
                directory.mkdir(parents=True)
                np.savez(directory / 'endpoint_counts.npz', endpoint=np.zeros((417, 2), np.int64))
            selection = dict(status='development_selected_real_evaluation_not_run', tie_order=list(e.ORDER),
                scores={a: 1.0 for a in e.ORDER}, results=results, winner='O', selected_at=1.0,
                freeze_manifest_sha256=e.sha(frozen / 'manifest.json'),
                checkpoint_sha256={a: e.sha(p) for a, p in ckpts.items()})
            save(root / 'development_selection.json', selection)
            r = NS(FROZEN=frozen, check_frozen=lambda **kw: {'reference_means': {'single': 1}},
                checked_run=lambda arm: (ckpts[arm], histories[arm]), check_gate=Mock(),
                select_winner=lambda scores: min(e.ORDER, key=scores.get))
            dg = NS(r3=NS(LEDGER_KEYS=('endpoint',), dev_score=lambda means, refs: means['single']),
                    r2=NS(MODES=('single',)), checked_parent=lambda: (results['O'], None, ckpts['O']))
            with patch.object(e, 'O_SHA', e.sha(ckpts['O'])):
                self.assertEqual(e.checked_selection(r, dg)[0]['winner'], 'O')
                selection['winner'] = 'B'
                save(root / 'development_selection.json', selection)
                with self.assertRaisesRegex(RuntimeError, 'Posthoc'):
                    e.checked_selection(r, dg)
                histories['C'][416]['source_render_diag']['matched_sampling_sha256'] = 'b' * 64
                with self.assertRaisesRegex(RuntimeError, 'Unmatched'):
                    e.checked_selection(r, dg)


class LiveMetadataTests(unittest.TestCase):
    def test_actual_parent_loader_paths_and_same_id_whitepoint_change(self):
        # Compile only the parent camera_rows AST; no torch or data imports.
        import ast
        parent = PATH.parents[1] / 'code/frozen/domislovic_v2.py'
        tree = ast.parse(parent.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'camera_rows')
        namespace = {'Path': Path, 'json': json, 'np': np, 'image_suffixes': lambda count: ['1', '12']}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(parent), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'explicit-data-root'
            camera = data / e.CAMERA
            split = {'two_illum_test': ['MockScene']}
            meta = {'MockScene': {'NumOfLights': 2, 'Light1': [1, 1, 1], 'Light2': [2, 1, .5]}}
            save(camera / 'split.json', split)
            save(camera / 'meta.json', meta)
            frozen = root / 'frozen'
            hashes = {name + '_sha256': e.sha(camera / (name + '.json')) for name in ('split', 'meta')}
            save(frozen / 'sources.json', hashes)
            r, dg = NS(FROZEN=frozen), NS(DATA=data)
            self.assertEqual(e.checked_live_metadata(r, dg), hashes)
            rows = namespace['camera_rows'](dg.DATA, e.CAMERA)
            self.assertTrue(all(Path(row['tiff']).parent == camera / 'test' for row in rows))
            meta['MockScene']['Light2'] = [4, 1, .25]
            save(camera / 'meta.json', meta)
            changed = namespace['camera_rows'](dg.DATA, e.CAMERA)
            self.assertEqual([r['id'] for r in rows], [r['id'] for r in changed])
            self.assertFalse(np.array_equal(rows[1]['chroma'], changed[1]['chroma']))
            with self.assertRaisesRegex(RuntimeError, 'metadata changed: meta'):
                e.checked_live_metadata(r, dg)

    def test_split_changed_without_image_id_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            camera = root / 'data' / e.CAMERA
            save(camera / 'split.json', {'two_illum_test': ['MockScene']})
            save(camera / 'meta.json', {'mock': True})
            frozen = root / 'frozen'
            save(frozen / 'sources.json', {name + '_sha256': e.sha(camera / (name + '.json')) for name in ('split', 'meta')})
            r, dg = NS(FROZEN=frozen), NS(DATA=root / 'data')
            e.checked_live_metadata(r, dg)
            # Same parsed content but different frozen bytes is also forbidden.
            with (camera / 'split.json').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(RuntimeError, 'metadata changed: split'):
                e.checked_live_metadata(r, dg)


class DeltaTests(unittest.TestCase):
    def test_18_cells_use_parent_aggregator_on_mock_evidence(self):
        analysis = PATH.parents[1] / 'analysis'
        import sys
        sys.path.insert(0, str(analysis))
        import compare_frozen_means as common
        import compare_round3_means as supports
        import evaluate_dg as checks
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ids = {'val': ['Place1_1', 'Place1_12'], 'test': ['Place2_1', 'Place2_12']}
            context = {'selection': {'winner': 'B'}, 'bindings': {'checkpoint_sha256': {a: a for a in e.MODELS}}}
            report = {'splits': {}}
            for subset in ids:
                report[subset + '_image_ids'] = ids[subset]
                report['splits'][subset] = {}
                for model, value in zip(e.MODELS, (4.0, 3.0, 2.0, 1.0, 2.0)):
                    summary = dict(model=model, split=subset, n_images=2, checkpoint_sha256=model,
                        patch_pooled={'mean': value}, patch_single={'mean': value}, patch_multi={'mean': value},
                        pixel_count=8, pixel_image_balanced_mean=value, pixel_pooled_mean=value)
                    report['splits'][subset][model] = summary
                    save(out / f'{subset}_{model}_summary.json', summary)
                    np.savez(out / f'{subset}_{model}_patches.npz', ids=np.array(ids[subset]),
                        patch_angular_error=np.full((2, 2), value, np.float32), valid_patch=np.ones((2, 2), bool))
                    with (out / f'{subset}_{model}_per_image.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['model', 'split', 'id', 'n_lights', 'valid_patches', 'valid_pixels', 'patch_mean', 'pixel_mean'])
                        writer.writeheader()
                        for i, image_id in enumerate(ids[subset]):
                            writer.writerow(dict(model=model, split=subset, id=image_id, n_lights=i + 1,
                                valid_patches=2, valid_pixels=4, patch_mean=value, pixel_mean=value))
            result = e.aggregate_results(out, context, report, common, supports, checks)
            self.assertEqual(result['metric_cells'], 18)
            self.assertEqual(len(result['metrics']['B']), 18)
            self.assertEqual(result['comparisons']['B']['passed_cells'], 18)
            self.assertEqual(result['outcome'], 'SEED0_GATE_MET')
            self.assertEqual(set(result['mechanism_delta_degrees']['B-A'].values()), {-1.0})

    def test_scene_grouping_and_sign_without_real_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ids = {'val': ['Place1_1', 'Place1_12', 'Place2_1'],
                   'test': ['Place3_1', 'Place3_12', 'Place4_1']}
            for subset in ids:
                for arm, value in (('A', 3.0), ('B', 2.0), ('C', 4.0)):
                    with (out / f'{subset}_{arm}_per_image.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['id', 'n_lights', 'valid_patches', 'valid_pixels', 'patch_mean', 'pixel_mean'])
                        writer.writeheader()
                        for i, image_id in enumerate(ids[subset]):
                            writer.writerow(dict(id=image_id, n_lights=2 if i == 1 else 1,
                                valid_patches=256, valid_pixels=65536, patch_mean=value, pixel_mean=value + .1))
            hashes = e.paired_deltas(out, ids)
            self.assertEqual(len(hashes), 2)
            with (out / 'per_scene_deltas.csv').open() as f:
                rows = list(csv.DictReader(f))
            row = next(r for r in rows if r['scene'] == 'Place1' and r['group'] == 'all' and r['contrast'] == 'B-A')
            self.assertEqual(int(row['compositions']), 2)
            self.assertEqual(float(row['patch_mean_delta']), -1)
            with self.assertRaises(FileExistsError):
                e.paired_deltas(out, ids)


if __name__ == '__main__':
    unittest.main()

"""CPU-only contract tests: no source image reads, training, or gate creation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / 'train_arms.py'
spec = importlib.util.spec_from_file_location('train_arms', FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class ContractTests(unittest.TestCase):
    def test_budget_is_exact_and_bounded(self):
        self.assertEqual(sum(runner.budget(c) for c in range(417)), 69600)
        self.assertEqual(runner.budget(416), 128)
        for bad in (-1, 417, 0.0):
            with self.assertRaises(RuntimeError): runner.budget(bad)

    def test_ties_preserve_preregistered_order(self):
        self.assertEqual(runner.select_winner(dict(O=1., A=1., B=1., C=1.)), 'O')
        self.assertEqual(runner.select_winner(dict(O=2., A=1., B=1., C=1.)), 'A')
        self.assertEqual(runner.select_winner(dict(O=2., A=2., B=1., C=1.)), 'B')
        for bad in (dict(A=1., O=1., B=1., C=1.), dict(O=1., A=float('nan'), B=1., C=1.)):
            with self.assertRaises(RuntimeError): runner.select_winner(bad)

    def test_missing_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'ROOT', Path(d)):
            with self.assertRaises(FileNotFoundError): runner.check_cache()

    def test_qc_cache_does_not_count_as_all_sources(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'ROOT', Path(d)):
            root = Path(d)
            (root / 'priors_manifest.json').write_text('{}')
            (root / 'cache_priors.py').write_text('fixture')
            (root / 'source_manifest.json').write_text(json.dumps({'source_count': 668, 'sources': []}))
            manifest = dict(schema_version=1, status='complete', source_manifest_sha256=runner.sha(root / 'source_manifest.json'), files={},priors_manifest_sha256=runner.sha(root/'priors_manifest.json'),cache_code_sha256=runner.sha(root/'cache_priors.py'))
            (root / 'cache_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, 'Source inventory'): runner.check_cache()

    def test_gate_never_materialized_by_check(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'FROZEN', Path(d)):
            with self.assertRaises(FileNotFoundError): runner.check_gate('formal', 'A')
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_evidence_hash_and_directory_enforced(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'ROOT', Path(d)):
            p = Path(d) / 'evidence.json'; p.write_text('{}')
            self.assertEqual(runner.evidence(dict(path=p.name, sha256=runner.sha(p))), p.resolve())
            with self.assertRaises(RuntimeError): runner.evidence(dict(path=p.name, sha256='0' * 64))

    def test_calibration_cannot_reuse_updated_full_weights(self):
        case = dict(steps=1, cycle=0, fresh_initialization=True, initial_sha256='initial', grad_norm=1., parameter_delta=1.,batch_indices=[0],consumed_views=8)
        done = dict(status='calibration_complete', arm='A', freeze_manifest_sha256='freeze',
                    cases={'full': case, 'tail': {**case, 'cycle': 416, 'fresh_initialization': False}}, outputs={})
        def digest(path): return 'initial' if Path(path).name == 'initial_seed0.pt' else 'freeze'
        with patch.object(runner, 'read', return_value=done), patch.object(runner, 'sha', side_effect=digest):
            with self.assertRaisesRegex(RuntimeError, 'separate fresh'): runner.check_calibration(Path('/unused/completion.json'), 'A')

    def test_cache_prior_and_code_provenance_fail_closed(self):
        source = dict(source_count=668,sources=[])
        base = dict(schema_version=1,status='complete',source_manifest_sha256='digest',priors_manifest_sha256='digest',cache_code_sha256='digest')
        for field in ('priors_manifest_sha256','cache_code_sha256'):
            for missing in (False,True):
                manifest = dict(base)
                if missing: del manifest[field]
                else: manifest[field] = 'stale'
                with patch.object(runner,'read',side_effect=[manifest,source]),patch.object(runner,'sha',return_value='digest'):
                    with self.assertRaisesRegex(RuntimeError,'provenance'): runner.check_cache()

    def test_calibration_rejects_full_cycle_consumption(self):
        case = dict(steps=1,cycle=0,fresh_initialization=True,initial_sha256='initial',grad_norm=1.,parameter_delta=1.,batch_indices=[0],consumed_views=1336)
        done = dict(status='calibration_complete',arm='A',freeze_manifest_sha256='freeze',cases={'full':case,'tail':dict(case,cycle=416)})
        def digest(path): return 'initial' if Path(path).name == 'initial_seed0.pt' else 'freeze'
        with patch.object(runner,'read',return_value=done),patch.object(runner,'sha',side_effect=digest):
            with self.assertRaisesRegex(RuntimeError,'consumption mismatch'): runner.check_calibration(Path('/unused/completion.json'),'A')

    def test_cache_inventory_must_equal_frozen_parent(self):
        rows = [dict(scene_id=str(i),sha256=str(i)) for i in range(668)]
        manifest = dict(schema_version=1,status='complete',source_manifest_sha256='digest',priors_manifest_sha256='digest',cache_code_sha256='digest')
        source = dict(source_count=668,sources=rows)
        parent = dict(source_ids=[r['scene_id'] for r in rows],raw_sha256={r['scene_id']:r['sha256'] for r in rows})
        parent['raw_sha256']['0'] = 'different'
        with patch.object(runner,'read',side_effect=[manifest,source,parent]),patch.object(runner,'sha',return_value='digest'):
            with self.assertRaisesRegex(RuntimeError,'frozen parent'): runner.check_cache()

if __name__ == '__main__': unittest.main()

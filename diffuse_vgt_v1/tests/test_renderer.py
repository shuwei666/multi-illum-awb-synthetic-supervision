import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('diffuse_renderer', Path(__file__).resolve().parents[1] / 'render_diffuse.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class RendererTests(unittest.TestCase):
    def inputs(self, dtype=np.float64):
        rng = np.random.default_rng(13)
        n = rng.normal(size=(32, 32, 3)).astype(dtype)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return dict(W=rng.uniform(.1, 1000000, (32, 32, 3)).astype(dtype),
                    s0=rng.uniform(.25, 4, (32, 32)).astype(dtype), normals=n,
                    technical_valid=np.ones((32, 32), bool),
                    lights=np.array([[2, 1, .5], [.5, 1, 2]], dtype=dtype),
                    directions=np.array([[0, 0, 1], [1, 0, 0]], dtype=dtype),
                    powers=np.array([1, 1], dtype=dtype), facing_normal=[0, 0, 1])

    def test_numerical_identities_both_precisions(self):
        for dtype, tolerance in ((np.float32, 1e-5), (np.float64, 1e-9)):
            result = renderer.render_diffuse(**self.inputs(dtype))
            for arm in result['arms'].values():
                self.assertEqual(arm['I'].dtype, dtype)
                relative = np.max(np.abs(arm['I'] / arm['E'] - arm['J']) / np.maximum(np.abs(arm['J']), 1e-15))
                self.assertLess(relative, tolerance)
                self.assertLess(np.max(np.abs(arm['E'][..., 1] - 1)), 1e-6)
                self.assertLessEqual(np.max(arm['I'] / 4), 60000)
            np.testing.assert_array_equal(result['arms']['A']['E'], result['arms']['B']['E'])
            np.testing.assert_array_equal(result['arms']['C']['H'], np.rot90(result['arms']['A']['H'], 2))
            self.assertEqual(np.log2(result['exposure']), int(np.log2(result['exposure'])))

    def test_common_power_before_after_normalization(self):
        for dtype in (np.float32, np.float64):
            inputs = self.inputs(dtype)
            first = renderer.render_diffuse(**inputs)
            inputs['powers'] *= 4
            second = renderer.render_diffuse(**inputs)
            np.testing.assert_array_equal(second['H_raw'], first['H_raw'] * 4)
            np.testing.assert_array_equal(second['T_raw'], first['T_raw'] * 4)
            for arm in first['arms']:
                for field in ('H', 'E', 'I', 'J'):
                    np.testing.assert_array_equal(second['arms'][arm][field], first['arms'][arm][field])

    def test_same_color_and_material_boundary(self):
        inputs = self.inputs()
        inputs['lights'][1] = inputs['lights'][0]
        same = renderer.render_diffuse(**inputs)
        np.testing.assert_allclose(same['arms']['A']['E'], np.broadcast_to(inputs['lights'][0], (32, 32, 3)), atol=1e-14)
        inputs = self.inputs()
        inputs['normals'][:] = [0, 0, 1]
        inputs['W'][:16] = [1, 20, 4]
        result = renderer.render_diffuse(**inputs)
        np.testing.assert_array_equal(result['arms']['A']['E'][0], result['arms']['A']['E'][-1])

    def test_orientation_relative_power_and_basis(self):
        inputs = self.inputs()
        inputs['normals'][:16] = [0, 0, 1]
        inputs['normals'][16:] = [1, 0, 0]
        first = renderer.render_diffuse(**inputs)
        self.assertGreater(first['u'][0, 0, 0], first['u'][-1, 0, 0])
        inputs['powers'] = [2, .5]
        second = renderer.render_diffuse(**inputs)
        self.assertGreater(np.max(np.abs(first['arms']['A']['E'] - second['arms']['A']['E'])), .1)
        inputs = self.inputs()
        baseline = renderer.render_diffuse(**inputs)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        inputs['normals'] = inputs['normals'] @ rotation.T
        inputs['directions'] = inputs['directions'] @ rotation.T
        transformed = renderer.render_diffuse(**inputs)
        np.testing.assert_allclose(baseline['H_raw'], transformed['H_raw'])

    def test_invalid_support_and_unmasked_patch_mean(self):
        inputs = self.inputs()
        inputs['normals'][0, 0] = np.nan
        inputs['s0'][1, 1] = 0
        result = renderer.render_diffuse(**inputs)
        self.assertFalse(result['technical_valid'][0, 0])
        self.assertFalse(result['technical_valid'][-1, -1])
        self.assertFalse(result['technical_valid'][1, 1])
        self.assertEqual(result['filled_s0'][1, 1], 1)
        labels, accepted = renderer.patch_labels(result, [[0, 0], [0, 16], [16, 0], [16, 16]], [True, False, True, True])
        np.testing.assert_array_equal(accepted, [False, False, True, False])
        for name, arm in result['arms'].items():
            self.assertTrue(np.isfinite(arm['I']).all())
            mean = arm['E'][:16, :16].mean(axis=(0, 1))
            np.testing.assert_allclose(labels[name][0], mean / np.linalg.norm(mean))

    def test_exposure_only_scales_image_and_neutral_target(self):
        inputs = self.inputs()
        first = renderer.render_diffuse(**inputs, storage_target=60000)
        second = renderer.render_diffuse(**inputs, storage_target=30000)
        self.assertEqual(second['exposure'], first['exposure'] / 2)
        for name in first['arms']:
            np.testing.assert_array_equal(second['arms'][name]['E'], first['arms'][name]['E'])
            for key in ('I', 'J'):
                np.testing.assert_array_equal(second['arms'][name][key], first['arms'][name][key] / 2)

    def test_fail_closed(self):
        for name, value in [('W', np.nan), ('lights', np.inf), ('powers', 0)]:
            inputs = self.inputs()
            inputs[name].flat[0] = value
            with self.assertRaises(ValueError):
                renderer.render_diffuse(**inputs)
        inputs = self.inputs()
        inputs['normals'] *= 2
        with self.assertRaises(ValueError):
            renderer.render_diffuse(**inputs)


if __name__ == '__main__':
    unittest.main()

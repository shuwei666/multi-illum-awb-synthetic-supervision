import copy
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preflight


class CompareTests(unittest.TestCase):
    def fixtures(self):
        parent = dict(draws={'order': 'same'}, indices='same', identity={'images': 'same'})
        arms = {}
        for i, name in enumerate(('A', 'B', 'C')):
            arms[name] = dict(parent, valid='v', ledger={'l': 'same'}, crop={'c': 'same'},
                lamps={'l': 'same'}, common_valid='v', probe_view_indices=[1],
                mixed_gt='shared' if name != 'C' else 'different',
                mixed_images=name, mixed_normalized_patches=name,
                normalized_probe_images=np.array([i], np.float32),
                normalized_probe_patches=np.array([i], np.float32))
        return parent, arms

    def test_matched_different_c_gt_allowed(self):
        self.assertEqual(set(preflight.compare(*self.fixtures())), {'A_B', 'A_C', 'B_C'})

    def test_mismatch_closed(self):
        for key in ('valid', 'ledger', 'crop', 'lamps', 'identity', 'draws', 'mixed_gt'):
            parent, arms = self.fixtures()
            arms['B'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                preflight.compare(parent, arms)

    def test_normalized_degeneracy_and_nonfinite_closed(self):
        for value in (0, float('nan')):
            parent, arms = self.fixtures()
            arms['B']['normalized_probe_images'] = np.array([value])
            with self.assertRaises(RuntimeError):
                preflight.compare(parent, arms)

    def test_array_digest_dtype_shape(self):
        a = np.array([1, 2], dtype=np.int64)
        self.assertNotEqual(preflight.digest_array(a), preflight.digest_array(a.reshape(1, 2)))
        self.assertNotEqual(preflight.digest_array(a), preflight.digest_array(a.astype(np.int32)))


if __name__ == '__main__':
    unittest.main()

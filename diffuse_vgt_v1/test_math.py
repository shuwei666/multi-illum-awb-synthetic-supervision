"""Synthetic algebra tests only; not a production renderer or prior validation."""
import json
import unittest
from pathlib import Path
import numpy as np

class AlgebraTests(unittest.TestCase):
    def test_inverse_decode(self):
        s=np.array([.25,1,4,20.])
        d=1/(1+s)
        np.testing.assert_allclose(1/np.clip(d,.001,1)-1,s,rtol=1e-12)
        self.assertEqual(float(1/np.clip(1.,.001,1)-1),0.)

    def test_three_arms_and_rotation(self):
        rng=np.random.default_rng(0)
        for dtype,tol in [(np.float64,1e-9),(np.float32,1e-5)]:
            w=rng.uniform(.1,2,(16,16,3)).astype(dtype)
            s=rng.uniform(.25,4,(16,16,1)).astype(dtype)
            h=rng.uniform(.1,2,(16,16,3)).astype(dtype)
            for base,field in [(w,h),(w/s,h),(w/s,h[::-1,::-1])]:
                e=field/field[:,:,1:2]
                i=base*field
                j=base*field[:,:,1:2]
                np.testing.assert_allclose(i/e,j,rtol=tol,atol=0)
                np.testing.assert_allclose(e[:,:,1],1,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(np.sort(h.ravel()),np.sort(h[::-1,::-1].ravel()))

    def test_common_power_normalization(self):
        rng=np.random.default_rng(1)
        u=rng.uniform(.1,2,(2,16,16,1))
        lights=np.array([[2,1,.5],[.7,1,1.8]])[:,None,None,:]
        raw=(u*lights).sum(0)
        t=u.sum(0)
        h=raw/np.median(t)
        np.testing.assert_allclose(7*raw/np.median(7*t),h,rtol=1e-12)
        np.testing.assert_allclose(raw/raw[:,:,1:2],h/h[:,:,1:2],rtol=1e-12)

    def test_same_color_and_orientation(self):
        normals=np.array([[0,0,1],[1,0,0.]])
        front=np.array([0,0,1.])
        u=.1+.9*np.maximum(normals@front,0)
        self.assertGreater(u[0],u[1])
        color=np.array([2,1,.5])
        h=u[:,None]*color
        np.testing.assert_allclose(h/h[:,1:2],np.broadcast_to(color,h.shape))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AlgebraTests))
    Path(__file__).with_name('test_report.json').write_text(json.dumps(dict(
        scope='Synthetic algebra only; no official uninvert comparison, real QC, priors or production input test',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        passed=result.wasSuccessful(),training_not_run=True),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())

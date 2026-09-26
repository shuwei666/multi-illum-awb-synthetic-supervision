"""Pure tensor tests; no source imagery, external model, or training."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ds, nr = load('diffuse_synthesis'), load('render_diffuse')


def crop_grid(sizes, left, top):
    # Parent round2 convention; test runner additionally checks actual parent.
    t = (torch.arange(512, device=sizes.device).float() + .5) / 512
    x = left[:, None] + sizes[:, None] * t
    y = top[:, None] + sizes[:, None] * t
    return torch.stack((x[:, None, :].expand(-1, 512, -1) / 256 - 1,
                        y[:, :, None].expand(-1, -1, 512) / 256 - 1), -1)


class TensorTests(unittest.TestCase):
    def test_capture_adapter_common_masks_identity_and_tail_accounting(self):
        n = 8
        wb = torch.full((n,3,256,256),20.,dtype=torch.float16)
        light = torch.tensor([2.,1.,.5]).expand(n,256,256,3).clone()
        capture = dict(wb=wb,branch=torch.tensor([0,2,0,2,0,2,0,2]),
                       patches=torch.arange(64).expand(n,-1),order=torch.arange(n-1,-1,-1),
                       consumed=torch.tensor([False,False,False,False,True,True,True,True]),
                       frameFloat=wb.float()*light.permute(0,3,1,2),light=light,
                       geom=[torch.zeros(n,dtype=torch.long),torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool)])
        prior = dict(s0=torch.full((n,1,256,256),2.),normal=torch.zeros(n,3,256,256),valid=torch.ones(n,1,256,256,dtype=torch.bool),invalid_fraction=torch.zeros(n))
        prior['normal'][:,2] = 1
        prior['valid'][:,:,0,0] = False
        endpoints = torch.tensor([[2.,1.,.5],[.5,1.,2.]]).expand(n,-1,-1)
        directions = torch.tensor([[0.,0.,1.],[1.,0.,0.]]).expand(n,-1,-1)
        powers = torch.ones(n,2)
        def extract(frame):
            return frame.reshape(n,3,16,16,16,16).permute(0,2,4,1,3,5).reshape(n,256,3,16,16)
        base = SimpleNamespace(EPS=1e-8,rot_flip=lambda x,*args:x,extract_patches=extract,zscore_bc=lambda x:x)
        results = [ds.adapt_capture(capture,prior,endpoints,directions,powers,base,mode=mode,chunk_size=3) for mode in ('A','B','C')]
        for mode, (chunked, mask) in zip(('A','B','C'), results):
            whole, wholemask = ds.adapt_capture(capture,prior,endpoints,directions,powers,base,mode=mode,chunk_size=n)
            self.assertTrue(torch.equal(mask,wholemask))
            for i in range(5):
                self.assertTrue(torch.equal(chunked[i],whole[i]),f'{mode} output {i} differs by chunking')
            self.assertEqual(chunked[-1],whole[-1])
        for result, common in results:
            self.assertEqual(int(common.sum()),n*63)
            self.assertEqual(result[-1]['consumed_valid_patches'],4*63)
            self.assertEqual(result[-1]['technical_rejected_patches'],n)
            self.assertTrue(torch.equal(result[3],results[0][0][3]))
            # Original branch unchanged even though priors request s0=2.
            identity_ordered = capture['branch'][capture['order']] == 0
            self.assertTrue(torch.equal(result[0][identity_ordered],capture['frameFloat'][capture['order']][identity_ordered]))

    def test_lamp_draws_and_stream_isolation(self):
        parent = torch.Generator().manual_seed(29)
        state = parent.get_state().clone()
        args = (1000, torch.eye(3), torch.Generator().manual_seed(90))
        d, a, params = ds.sample_lamps(*args)
        self.assertTrue(torch.equal(state, parent.get_state()))
        self.assertTrue(torch.allclose(d.norm(dim=-1), torch.ones(1000, 2), atol=1e-6))
        self.assertGreaterEqual(float(params['cos_theta'].min()), np.cos(np.deg2rad(70)))
        self.assertGreaterEqual(float((a[:, 0] / a[:, 1]).min()), .5)
        self.assertLessEqual(float((a[:, 0] / a[:, 1]).max()), 2)
        d2, a2, _ = ds.sample_lamps(1000, torch.eye(3), torch.Generator().manual_seed(90))
        self.assertTrue(torch.equal(d, d2) and torch.equal(a, a2))

    def test_numpy_equivalence_and_common_power(self):
        rng = np.random.default_rng(12)
        W = rng.uniform(1, 100000, (32, 32, 3)).astype('float32')
        normals = rng.normal(size=W.shape).astype('float32')
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        s0 = rng.uniform(.25, 4, (32, 32)).astype('float32')
        endpoints = np.array([[2, 1, .6], [.5, 1, 1.8]], 'float32')
        directions = np.array([[0, 0, 1], [1, 0, 0]], 'float32')
        powers = np.array([1.2, .8], 'float32')
        numpy_result = nr.render_diffuse(W,s0,normals,np.ones((32,32),bool),endpoints,directions,powers,facing_normal=[0,0,1],storage_divisor=1)
        wb = torch.from_numpy(W).permute(2,0,1)[None]
        prior = dict(s0=torch.from_numpy(s0)[None,None],normal=torch.from_numpy(normals).permute(2,0,1)[None])
        args = (wb,prior,torch.from_numpy(endpoints)[None],torch.from_numpy(directions)[None],torch.from_numpy(powers)[None])
        result = ds.render_fields(*args)
        brighter = ds.render_fields(*args[:-1],args[-1]*4)
        for name in ('A','B','C'):
            for key in ('H','E','I','J'):
                np.testing.assert_allclose(result['arms'][name][key][0].permute(1,2,0).numpy(),numpy_result['arms'][name][key],rtol=2e-6,atol=1e-5)
                self.assertTrue(torch.equal(result['arms'][name][key],brighter['arms'][name][key]))
            arm = result['arms'][name]
            self.assertTrue(torch.allclose(arm['I']/arm['E'],arm['J'],rtol=1e-5,atol=1e-5))

    def test_conservative_validity_and_cancelling_normals(self):
        cache = dict(s0=torch.ones(1,1,512,512),normal=torch.zeros(1,3,512,512),valid=torch.ones(1,1,512,512,dtype=torch.bool))
        cache['normal'][:,2] = 1
        cache['valid'][0,0,10,10] = False
        cache['shading_valid'] = cache['valid'].clone()
        cache['normal_valid'] = cache['valid'].clone()
        cache['normal'][0,:,20,20] = torch.tensor([1.,0,0])
        cache['normal'][0,:,20,21] = torch.tensor([-1.,0,0])
        cache['normal'][0,:,21,20] = torch.tensor([1.,0,0])
        cache['normal'][0,:,21,21] = torch.tensor([-1.,0,0])
        empty = dict(chosen=torch.empty(0,dtype=torch.long),sizes=torch.empty(0),left=torch.empty(0),top=torch.empty(0))
        result = ds.resample_priors(cache,empty,crop_grid,torch.tensor([0.,0,1]))
        self.assertFalse(bool(result['normal_valid'][0,0,5,5]))
        self.assertFalse(bool(result['normal_valid'][0,0,10,10]))
        self.assertFalse(bool(result['valid'][0,0,250,250]))
        self.assertTrue(torch.isfinite(result['normal']).all())
        self.assertTrue(torch.isfinite(result['s0']).all())

    def test_crop_replay_support_and_patch_mean(self):
        g1, g2 = torch.Generator().manual_seed(77), torch.Generator().manual_seed(77)
        p1, p2 = ds.replay_crop_plan(8,'cpu',g1), ds.replay_crop_plan(8,'cpu',g2)
        self.assertTrue(all(torch.equal(p1[k],p2[k]) for k in p1))
        field = torch.arange(256*256).reshape(1,1,256,256).float().expand(-1,3,-1,-1)
        means = ds.patch_mean(field)
        self.assertEqual(float(means[0,0,0]),float(field[0,0,:16,:16].mean()))
        self.assertEqual(float(means[0,17,0]),float(field[0,0,16:32,16:32].mean()))
        cache = dict(s0=torch.ones(1,1,512,512),normal=torch.zeros(1,3,512,512),valid=torch.ones(1,1,512,512,dtype=torch.bool))
        cache['normal'][:,2] = 1
        cache['valid'][0,0,200,200] = False
        cache['shading_valid'] = cache['valid'].clone()
        cache['normal_valid'] = cache['valid'].clone()
        plan = dict(chosen=torch.tensor([0]),sizes=torch.tensor([385]),left=torch.tensor([20]),top=torch.tensor([30]))
        result = ds.resample_priors(cache,plan,crop_grid,torch.tensor([0.,0,1]))
        grid = crop_grid(plan['sizes'],plan['left'],plan['top'])
        invalid_mass = F.avg_pool2d(F.grid_sample((~cache['valid']).float(),grid,align_corners=False,padding_mode='border'),2)
        self.assertTrue(torch.equal(result['normal_valid'][:1],invalid_mass==0))

    def test_shading_invalidity_is_not_rotated(self):
        valid = torch.ones(1,1,512,512,dtype=torch.bool)
        cache = dict(s0=torch.ones(1,1,512,512),normal=torch.zeros(1,3,512,512),shading_valid=valid.clone(),normal_valid=valid.clone())
        cache['normal'][:,2] = 1
        cache['shading_valid'][0,0,10,10] = False
        empty = dict(chosen=torch.empty(0,dtype=torch.long),sizes=torch.empty(0),left=torch.empty(0),top=torch.empty(0))
        result = ds.resample_priors(cache,empty,crop_grid,torch.tensor([0.,0,1]))
        self.assertFalse(bool(result['valid'][0,0,5,5]))
        self.assertTrue(bool(result['valid'][0,0,250,250]))
        self.assertTrue(bool(result['normal_valid'].all()))
        # A normal-invalid pixel must still reject both local and rotated support.
        cache['normal_valid'][0,0,10,10] = False
        result = ds.resample_priors(cache,empty,crop_grid,torch.tensor([0.,0,1]))
        self.assertFalse(bool(result['valid'][0,0,250,250]))

    def test_one_batch_calibration_ledger_full_and_tail(self):
        n = 1336
        order = torch.arange(n-1,-1,-1)
        capture = dict(order=order, branch=torch.tensor([0,2]).repeat(n//2),
                       first=torch.arange(n//2),second=torch.arange(n//2)+1)
        common = torch.ones(n,64,dtype=torch.bool)
        common[:,0] = False
        for index in (0,127):
            capture['consumed'] = ds.select_consumed_batches(order,[index])
            self.assertEqual(int(capture['consumed'].sum()),8)
            self.assertTrue(bool(capture['consumed'][order[index*8:(index+1)*8]].all()))
            ledger, patches, views = ds.consumption_ledger(capture,common,1353)
            self.assertEqual((patches,views),(4*63,4))
            for name,value in ledger.items():
                self.assertEqual(int(value.sum()),patches if 'patch' in name else views)
        for indices in ([],[0,0],[-1],[167]):
            with self.assertRaises(ValueError): ds.select_consumed_batches(order,indices)


if __name__ == '__main__':
    unittest.main()

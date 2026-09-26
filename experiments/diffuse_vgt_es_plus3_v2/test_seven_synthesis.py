"""CPU arithmetic tests; separate from actual 668-source full/tail preflight."""
import unittest
import torch
from seven_synthesis import fields, ARMS, Adapter

class TestSeven(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.W=torch.rand(2,3,256,256)*90000
        self.s=torch.rand(2,1,256,256)*3.75+.25
        self.H=torch.rand(2,3,256,256)+.2
        self.H[:,1:2]=torch.rand(2,1,256,256)+.2
        self.producer=lambda *a,**k:dict(arms={'A':dict(H=self.H)},H_raw=self.H)
        self.out=fields(self.producer,self.W,dict(s0=self.s),None,None,None)

    def test_inventory(self): self.assertEqual(set(self.out['arms']),set(ARMS))
    def test_formulas_and_exposure(self):
        H=self.H; S=H[:,1:2]; E=H/S; W=self.W; R=W/self.s
        q=lambda x:torch.flip(x,(-2,-1))
        expected={'ES00':(R*H,E,R*S),'ES11':(R*q(H),q(E),R*q(S)), 'ES10':(R*S*q(E),q(E),R*S),'ES01':(R*q(S)*E,E,R*q(S)), 'KEEP_OLD_R':(W*q(H),q(E),W*q(S)), 'SHAM_OLD_0':(W/q(self.s)*H,E,W/q(self.s)*S),'NO_NEW_S':(R*E,E,R)}
        peak=torch.stack([v[0].flatten(1).amax(1) for v in expected.values()]+[(W*H).flatten(1).amax(1)]).amax(0)
        f=2**(-torch.ceil(torch.log2((peak/60000).clamp_min(1))))
        self.assertTrue(torch.equal(f,self.out['exposure']))
        for name,(I,label,J) in expected.items():
            arm=self.out['arms'][name]
            self.assertTrue(torch.equal(arm['I'],I*f[:,None,None,None]))
            self.assertTrue(torch.equal(arm['E'],label))
            self.assertTrue(torch.equal(arm['J'],J*f[:,None,None,None]))
            self.assertTrue(torch.allclose(arm['I']/arm['E'],arm['J'],rtol=3e-6,atol=.02))
            self.assertLessEqual(float(arm['I'].max()),60000)
    def test_labels(self):
        for group in (('ES00','ES01','SHAM_OLD_0','NO_NEW_S'),('ES10','ES11','KEEP_OLD_R')):
            for name in group:
                self.assertTrue(torch.equal(self.out['arms'][group[0]]['E'],self.out['arms'][name]['E']))
    def test_sham_preserves_multiset(self):
        self.assertTrue(torch.equal(self.s.flatten().sort().values,torch.flip(self.s,(-2,-1)).flatten().sort().values))
    def test_no_rng_consumed(self):
        before=torch.get_rng_state().clone()
        fields(self.producer,self.W,dict(s0=self.s),None,None,None)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_rotated_shading_validity(self):
        class Synth:
            render_fields=lambda *a,**k:None
            resample_priors=lambda *a,**k:dict(valid=torch.ones(1,1,256,256,dtype=torch.bool),shading_valid=torch.ones(1,1,256,256,dtype=torch.bool))
        adapter=Adapter(Synth())
        old=adapter.original_resample
        def fake(*a,**k):
            p=old(); p['shading_valid'][0,0,0,0]=False; p['valid'][0,0,0,0]=False
            return p
        adapter.original_resample=fake
        p=adapter.resample()
        self.assertFalse(bool(p['valid'][0,0,0,0]))
        self.assertFalse(bool(p['valid'][0,0,-1,-1]))
        self.assertEqual(int((~p['valid']).sum()),2)

if __name__=='__main__': unittest.main()

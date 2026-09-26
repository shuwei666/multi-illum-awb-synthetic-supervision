"""Actual normalized global/local intervention probes; no optimizer updates."""
import json
import torch
import queue_train as q
from seven_synthesis import ARMS

def main():
    q.check_frozen(True)
    out=q.ROOT/'input_signal.json'
    assert not out.exists()
    dg,synth=q.runtime()
    with dg.r3.exclusive_lock():
        dg.r3.gpu_preflight()
        source,pool,bank,cache,basis=q.parent.load_inputs()
        cases={}
        for cycle in (0,416):
            stats={}; reference=None
            for arm in ARMS:
                capture={}
                rendered=synth.render_cycle(source,pool,bank,q.parent.CONFIG,cycle,0,q.parent.budget(cycle),cache=cache,basis=basis,mode=arm,capture=capture)
                c=capture['parent']; selected=torch.where(c['branch'][c['order']]==2)[0][:16]
                image=rendered[0][selected].float()
                patches=rendered[1].reshape(len(rendered[0]),64,3,16,16)[selected].reshape(-1,3,16,16)
                local=dg.base.zscore_bc(patches.float())
                if reference is None:reference=(image.clone(),local.clone())
                stats[arm]=dict(global_normalized_rms_vs_ES00=float((image-reference[0]).square().mean().sqrt()),
                    local_normalized_rms_vs_ES00=float((local-reference[1]).square().mean().sqrt()),
                    sampled_mixed_views=len(selected),sampling_sha256=rendered[5]['matched_sampling_sha256'],
                    exposure_min=rendered[5]['exposure_min'],technical_rejected_patches=rendered[5]['technical_rejected_patches'])
                assert all(torch.isfinite(x).all() for x in (image,local))
                del rendered,capture,c,image,local,patches
            cases[str(cycle)]=stats
        q.write(out,dict(status='PASS',freeze_manifest_sha256=q.sha(q.FROZEN/'manifest.json'),
            scope='actual_full_tail_first16mixed_views_after_global_and_local_normalization',cases=cases))

if __name__=='__main__':main()

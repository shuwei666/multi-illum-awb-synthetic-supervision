"""Full-cohort matched render diagnostic; no optimizer, dev, or test access.

PASS means the executable assertions passed, never independent approval.
Run only after frozen code/cache and independent permission for this diagnostic.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
from pathlib import Path
import time

import train_arms as runner


def digest_array(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def tensor_digest(tensor, indices=None, transform=None, chunk=32):
    """Bound temporary device/host copies, preserving original dtype and order."""
    import torch
    indices = torch.arange(len(tensor), device=tensor.device) if indices is None else indices
    h = hashlib.sha256()
    h.update(str((len(indices), tuple(tensor.shape[1:]))).encode())
    for start in range(0, len(indices), chunk):
        part = tensor[indices[start:start + chunk]]
        if transform is not None:
            part = transform(part.float())
        runner.require(bool(torch.isfinite(part).all()), 'Nonfinite digest input')
        h.update(str(part.dtype).encode())
        h.update(part.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def snapshot(result, capture, base):
    import torch
    images, patches, gt, valid, idx, diag, ledger = result
    p = capture.get('parent', capture)
    ordered_branch = p['branch'][p['order']]
    identity = torch.where(ordered_branch == 0)[0]
    mixed = torch.where(ordered_branch == 2)[0]
    patch_identity = torch.where(ordered_branch.repeat_interleave(64) == 0)[0]
    patch_mixed = torch.where(ordered_branch.repeat_interleave(64) == 2)[0]
    sample_views = mixed[:16]
    sample_patches = (sample_views[:, None] * 64 + torch.arange(64, device=idx.device)).flatten()
    draws = {k: tensor_digest(p[k]) for k in ('first', 'second', 'patches', 'branch', 'order', 'consumed', 'wb')}
    draws.update({f'geom{i}': tensor_digest(v) for i, v in enumerate(p['geom'])})
    out = dict(draws=draws, indices=tensor_digest(idx), valid=tensor_digest(valid),
        ledger={k: digest_array(v) for k, v in ledger.items()},
        identity={k: tensor_digest(t, sel) for k, t, sel in (
            ('images', images, identity), ('patches', patches, patch_identity), ('gt', gt, patch_identity))},
        mixed_gt=tensor_digest(gt, patch_mixed),
        mixed_images=tensor_digest(images, mixed),
        mixed_normalized_patches=tensor_digest(patches, patch_mixed, base.zscore_bc),
        diag=diag,
        normalized_probe_images=images[sample_views].float().cpu().numpy(),
        normalized_probe_patches=base.zscore_bc(patches[sample_patches].float()).cpu().numpy(),
        probe_view_indices=sample_views.cpu().tolist(),
        probe_patch_count=len(sample_patches))
    if 'crop_plan' in capture:
        out['crop'] = {k: tensor_digest(v) for k, v in capture['crop_plan'].items()}
        out['lamps'] = {k: tensor_digest(capture[k]) for k in ('directions', 'powers')}
        out['common_valid'] = tensor_digest(capture['common_valid'])
    return out


def compare(parent, arms):
    import numpy as np
    runner.require(set(arms) == {'A', 'B', 'C'}, 'Incomplete arm inventory')
    for name, arm in arms.items():
        runner.require(arm['draws'] == parent['draws'] and arm['indices'] == parent['indices'], f'{name}: parent sampling changed')
        runner.require(arm['identity'] == parent['identity'], f'{name}: parent identity changed')
    for key in ('draws', 'indices', 'valid', 'ledger', 'crop', 'lamps', 'common_valid', 'probe_view_indices'):
        runner.require(arms['A'][key] == arms['B'][key] == arms['C'][key], f'Unmatched {key}')
    runner.require(arms['A']['mixed_gt'] == arms['B']['mixed_gt'], 'A/B mixed GT changed')
    differences = {}
    for a, b in itertools.combinations(arms, 2):
        item = {}
        for kind in ('images', 'patches'):
            key = 'normalized_probe_' + kind
            delta = arms[a][key].astype(np.float64) - arms[b][key].astype(np.float64)
            rms = float(np.sqrt(np.mean(delta * delta)))
            runner.require(np.isfinite(rms) and rms > 0, f'{a}/{b}: no finite normalized {kind} intervention')
            item[kind + '_probe_rms'] = rms
        runner.require(arms[a]['mixed_images'] != arms[b]['mixed_images'] and
            arms[a]['mixed_normalized_patches'] != arms[b]['mixed_normalized_patches'], 'Whole mixed stream unchanged')
        differences[a + '_' + b] = item
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    runner.require(not args.output.exists(), 'Never overwrite a preflight result')
    frozen = runner.check_frozen(check_sources=True)
    import torch
    dg, synth = runner.runtime()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    output = dict(status='RUNNING', scope='render_preflight_only', seed=0,
        freeze_manifest_sha256=runner.sha(runner.FROZEN / 'manifest.json'),
        preflight_code_sha256=runner.sha(Path(__file__)), code_hashes=runner.code_hashes(),
        cache_manifest_sha256=runner.sha(runner.ROOT / 'cache_manifest.json'), cases={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        source, pool, bank, cache, basis = runner.load_inputs()
        runner.require(len(source[1]) == 668 and len(pool) == 1353, 'Wrong cohort')
        for label, cycle, steps in (('full', 0, 167), ('tail', 416, 128)):
            case_start = time.monotonic()
            pc = {}
            result = dg.synth.render_cycle(source, pool, bank, runner.CONFIG, cycle, 0, steps, mode='O', capture=pc)
            parent = snapshot(result, pc, dg.base)
            del result, pc
            gc.collect(); torch.cuda.empty_cache()
            arms = {}
            for arm in runner.ARMS:
                cap = {}
                result = synth.render_cycle(source, pool, bank, runner.CONFIG, cycle, 0, steps,
                    cache=cache, basis=basis, mode=arm, capture=cap)
                runner.validate_render(result, arm, steps)
                arms[arm] = snapshot(result, cap, dg.base)
                runner.require(int(cap['parent']['consumed'].sum()) == steps * 8, 'Wrong full/tail consumption')
                del result, cap
                gc.collect(); torch.cuda.empty_cache()
            difference = compare(parent, arms)
            for record in (parent, *arms.values()):
                del record['normalized_probe_images'], record['normalized_probe_patches']
            torch.cuda.synchronize()
            output['cases'][label] = dict(cycle=cycle, consumed_steps=steps, parent=parent,
                arms=arms, pairwise_normalized_input_difference=difference,
                seconds=time.monotonic() - case_start)
            runner.write(args.output, output)
        runner.require(runner.sha(runner.FROZEN / 'manifest.json') == output['freeze_manifest_sha256']
            and runner.code_hashes() == output['code_hashes'], 'Code/freeze changed while executing')
        output['status'] = 'PASS'
    except Exception as exc:
        output.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        output.update(seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            interpretation='Executable assertions only; not independent approval or performance evidence. RMS probes use first 16 mixed ordered views; full mixed streams also hashed.')
        runner.write(args.output, output)


if __name__ == '__main__':
    main()

"""Fixed 24 analytic Lambertian observations; frozen-prior inference only.

Generate/selfcheck run on CPU with NumPy only. Infer reuses qc_code_002.py's
existing CUDA loaders, with strict asset checks and no download fallback.
These simplified RGB scenes do not validate physical accuracy on Nikon images.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

SIZE = 512
ETA = 0.1
GEOMETRIES = ('plane', 'fold', 'sphere')
LAYOUTS = ('checker', 'stripes')
LIGHTS = np.array([[0, 0, 1], [.6, 0, .8], [-.6, 0, .8], [0, .6, .8]], np.float32)
RELIGHT_DIRECTIONS = np.array([[.6, 0, .8], [-.3, .4, np.sqrt(.75)]], np.float32)
RELIGHT_COLORS = np.array([[1.3, 1, .7], [.7, 1, 1.3]], np.float32)
RELIGHT_POWERS = np.array([1, 1], np.float32)
EXPECTED_LOADER_SHA256 = '8f6a6c6d2ab4aaa3ff8f63feb029838b19fb18e881889c1e1a87f7675597304f'
LIMITATIONS = [
    '24 observations of six geometry/material layouts, not 24 independent scenes.',
    'Analytic orthographic normals, diffuse RGB reflectance and directional illumination only.',
    'No cast shadows, interreflection, spectral camera response, gloss, transparency or sensor noise.',
    'DSINE uses inherited approximate FOV60 perspective intrinsics on orthographic images.',
    'The synthetic RGB domain and material patterns are out of the Nikon photographic domain.',
    'Normal coordinates are right/down/front (+Z); no fitted rotation or sign optimization.',
    'Material association is a diagnostic correlation, not proof of intrinsic decomposition.',
    'No model, threshold, light parameter or AWB protocol may be changed from these results.',
    'Inference completion is not an independent review PASS or Nikon physical accuracy claim.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def contract():
    return dict(version=2, size=SIZE, eta=ETA, geometries=GEOMETRIES,
                material_layouts=LAYOUTS, source_lights=LIGHTS.tolist(),
                relight_directions=RELIGHT_DIRECTIONS.tolist(),
                relight_colors=RELIGHT_COLORS.tolist(), relight_powers=RELIGHT_POWERS.tolist(),
                illumination='eta + (1-eta)*max(dot(normal, direction),0); source light neutral RGB',
                input='reflectance * true_shading; max-normalized proxy for both priors',
                shading_adapter='uninvert -> bilinear resize -> median normalization -> clip[.25,4]',
                metrics='natural-log scale alignment; pixel-unweighted mean within declared masks',
                limitations=LIMITATIONS, awb_training=False)


def geometry(name):
    axis = (np.arange(SIZE, dtype=np.float32) + .5) / SIZE * 2 - 1
    x, y = np.meshgrid(axis, axis)
    n = np.zeros((SIZE, SIZE, 3), np.float32)
    mask = np.ones((SIZE, SIZE), bool)
    if name == 'plane':
        n[..., 2] = 1
    elif name == 'fold':
        n[..., 0] = np.where(x < 0, -.5, .5)
        n[..., 2] = np.sqrt(.75)
    elif name == 'sphere':
        # A visible convex sphere cap; invalid background remains black.
        r2 = x*x + y*y
        mask = r2 < .90**2
        n[..., 0] = x
        n[..., 1] = y
        n[..., 2] = np.sqrt(np.maximum(1-r2, 0))
        n[~mask] = [0, 0, 1]
    else:
        raise ValueError(name)
    return n, mask


def reflectance(layout):
    yy, xx = np.indices((SIZE, SIZE))
    pattern = ((xx//32 + yy//32) % 2) if layout == 'checker' else ((xx//24) % 2)
    # Non-proportional RGB materials with different neutral brightness.
    return np.where(pattern[..., None] == 0,
                    np.array([.18, .25, .32], np.float32),
                    np.array([.70, .50, .38], np.float32)).astype(np.float32)


def observations():
    for geom in GEOMETRIES:
        n, mask = geometry(geom)
        for layout in LAYOUTS:
            rho = reflectance(layout)
            for light_index, direction in enumerate(LIGHTS):
                s = (ETA + (1-ETA)*np.maximum(n @ direction, 0)).astype(np.float32)
                w = rho * s[..., None]
                w[~mask] = 0
                yield f'{geom}_{layout}_L{light_index}', dict(
                    input_linear=w, reflectance=rho, shading=s, normal=n,
                    support=mask, source_direction=direction.copy(),
                    relight_directions=RELIGHT_DIRECTIONS.copy(),
                    relight_colors=RELIGHT_COLORS.copy(), relight_powers=RELIGHT_POWERS.copy())


def scale_log_metrics(prediction, truth, mask):
    if not mask.any():
        raise RuntimeError('Empty metric support')
    lp, lt = np.log(prediction[mask].astype(np.float64)), np.log(truth[mask].astype(np.float64))
    delta = lp-lt
    offset = float(delta.mean())
    centered = delta-offset
    return dict(log_rmse=float(np.sqrt(np.mean(centered**2))),
                log_mae=float(np.mean(np.abs(centered))),
                log_scale_offset=offset, multiplicative_scale=float(np.exp(offset)))


def neutral_h(normals, mask):
    response = ETA + (1-ETA)*np.maximum(normals @ RELIGHT_DIRECTIONS.T, 0)
    h = (response * RELIGHT_POWERS) @ RELIGHT_COLORS
    return (h / np.median(h[..., 1][mask])).astype(np.float32)


def metric_support(support, shading_valid, normal_valid):
    mask = support & shading_valid & normal_valid
    if not mask.any():
        raise RuntimeError('Empty common prior support')
    return mask


def material_association(estimate, truth, rho, mask):
    # Adjacent log-gradients remove constant scale and directly test detail leakage.
    residual = np.log(estimate.astype(np.float64))-np.log(truth.astype(np.float64))
    material = np.log(rho.mean(2).astype(np.float64))
    error_grad, material_grad = [], []
    for axis in (0, 1):
        pair = mask[:-1, :] & mask[1:, :] if axis == 0 else mask[:, :-1] & mask[:, 1:]
        error_grad.append(np.diff(residual, axis=axis)[pair])
        material_grad.append(np.diff(material, axis=axis)[pair])
    e, r = np.concatenate(error_grad), np.concatenate(material_grad)
    ec, rc = e-e.mean(), r-r.mean()
    denominator = float(np.sqrt(np.sum(ec**2)*np.sum(rc**2)))
    return dict(adjacent_log_gradient_correlation=float(np.sum(ec*rc)/denominator) if denominator > 1e-12 else None,
                correlation_undefined_reason=None if denominator > 1e-12 else 'zero error or material gradient variance',
                material_gradient_projection_slope=float(np.sum(ec*rc)/np.sum(rc**2)) if np.sum(rc**2)>0 else None,
                residual_gradient_rms=float(np.sqrt(np.mean(e*e))), adjacent_pair_count=int(len(e)))


def evaluate(truth, estimated_s, estimated_n, shading_valid, normal_valid):
    support = truth['support']
    sm = support & shading_valid
    nm = support & normal_valid
    common = metric_support(support, shading_valid, normal_valid)
    if not sm.any() or not nm.any():
        raise RuntimeError('Empty shading or normal support')
    n64 = estimated_n.astype(np.float64)
    t64 = truth['normal'].astype(np.float64)
    dots = (n64*t64).sum(2)/(np.linalg.norm(n64,axis=2)*np.linalg.norm(t64,axis=2))
    ae = np.degrees(np.arccos(np.clip(dots[nm], -1, 1)))
    true_h = neutral_h(truth['normal'], common)
    estimated_h = neutral_h(estimated_n, common)
    # Common medians remove unidentifiable source-shading scale before relighting.
    true_s = truth['shading']/np.median(truth['shading'][common])
    est_s = estimated_s/np.median(estimated_s[common])
    w = truth['input_linear']
    reference = w/true_s[..., None]*true_h
    factorial = {}
    for name, old_s, new_h in [
        ('true_shading_true_normal', true_s, true_h),
        ('estimated_shading_true_normal', est_s, true_h),
        ('true_shading_estimated_normal', true_s, estimated_h),
        ('estimated_shading_estimated_normal', est_s, estimated_h),
    ]:
        relit = w/old_s[..., None]*new_h
        channel_mask = np.broadcast_to(common[..., None], w.shape)
        row = scale_log_metrics(relit, reference, channel_mask)
        p, t = relit[common].astype(np.float64), reference[common].astype(np.float64)
        color_angle = np.degrees(np.arccos(np.clip((p*t).sum(1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(t,axis=1)), -1, 1)))
        row['rgb_angle_mean_deg'] = float(color_angle.mean())
        # Raw intensity bias remains available; RGB angle cannot see neutral error.
        row['unaligned_relative_rgb_rmse'] = float(np.sqrt(np.mean((p-t)**2))/np.sqrt(np.mean(t*t)))
        row['unaligned_mean_luminance_ratio'] = float(p.mean()/t.mean())
        factorial[name] = row
    return dict(shading_raw=scale_log_metrics(estimated_s, truth['shading'], sm),
                normal_mean_deg=float(ae.mean()), normal_median_deg=float(np.median(ae)),
                normal_p95_deg=float(np.percentile(ae,95)),
                material_detail=material_association(estimated_s, truth['shading'], truth['reflectance'], sm),
                support_pixels=int(support.sum()), shading_valid_pixels=int(sm.sum()),
                normal_valid_pixels=int(nm.sum()), common_pixels=int(common.sum()),
                relighting_factorial=factorial)


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'truth_manifest.json'
    frozen_contract = contract()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['contract'] != json.loads(json.dumps(frozen_contract)):
            raise RuntimeError('Existing truth contract differs; use a fresh output directory')
        for row in manifest['observations']:
            if sha(output/row['file']) != row['sha256']:
                raise RuntimeError('Truth file hash mismatch: '+row['file'])
        if len(manifest['observations']) != 24:
            raise RuntimeError('Expected exactly 24 observations')
        return manifest
    rows = []
    (output/'truth').mkdir(exist_ok=True)
    for identifier, arrays in observations():
        path = output/'truth'/f'{identifier}.npz'
        if path.exists():
            # Partial generation may resume only with identical array content.
            with np.load(path, allow_pickle=False) as previous:
                if set(previous.files) != set(arrays) or any(array_sha(previous[k]) != array_sha(v) for k,v in arrays.items()):
                    raise RuntimeError('Existing truth differs: '+identifier)
        else:
            np.savez_compressed(path, **arrays)
        rows.append(dict(id=identifier, file=str(path.relative_to(output)), sha256=sha(path),
                         arrays={key:array_sha(value) for key,value in arrays.items()}))
    manifest = dict(contract=frozen_contract, observations=rows)
    save_json(manifest_path, manifest)
    return manifest


def verify_assets(old_root):
    manifest_path = old_root/'priors_manifest.json'
    priors = json.loads(manifest_path.read_text())
    if sha(old_root/'qc_code_002.py') != EXPECTED_LOADER_SHA256:
        raise RuntimeError('Inherited loader differs from reviewed qc_code_002.py')
    verified = {}
    for name, row in priors['weights'].items():
        path = old_root/'weights'/name
        if sha(path) != row['sha256']:
            raise RuntimeError('Weight hash mismatch: '+name)
        verified['weights/'+name] = row['sha256']
    missing_files = []
    for name, row in priors['repos'].items():
        for relative, digest in row['python_files'].items():
            path = old_root/'vendor'/name/relative
            if not path.exists():
                missing_files.append('vendor/'+name+'/'+relative)
                continue
            if sha(path) != digest:
                raise RuntimeError('Vendor hash mismatch: '+str(path))
            verified['vendor/'+name+'/'+relative] = digest
    if missing_files:
        raise RuntimeError('Missing frozen vendor manifest files: '+json.dumps(missing_files))
    return dict(priors_manifest_sha256=sha(manifest_path), files=verified,
                loader_sha256=sha(old_root/'qc_code_002.py'),
                renderer_sha256=sha(old_root/'render_diffuse.py'))


def selfcheck():
    count = 0
    for identifier, truth in observations():
        mask = truth['support']
        np.testing.assert_allclose(np.linalg.norm(truth['normal'][mask],axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(truth['input_linear'][mask],
                                   (truth['reflectance']*truth['shading'][...,None])[mask], rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(truth['source_direction']),1,atol=1e-6)
        if not np.all(truth['shading'] > 0):
            raise RuntimeError('Nonpositive truth shading')
        metrics = evaluate(truth, truth['shading'], truth['normal'], mask, mask)
        assert metrics['shading_raw']['log_rmse'] < 1e-7
        assert metrics['normal_mean_deg'] < 1e-5
        for row in metrics['relighting_factorial'].values():
            assert row['log_rmse'] < 1e-7 and row['unaligned_relative_rgb_rmse'] < 1e-7
        scaled = scale_log_metrics(truth['shading']*3, truth['shading'], mask)
        assert scaled['log_rmse'] < 1e-6
        # Deliberate material leakage must be detected, including on neutral planes.
        material = truth['reflectance'].mean(2)
        association = material_association(truth['shading']*material,truth['shading'],truth['reflectance'],mask)
        assert association['adjacent_log_gradient_correlation'] > .99999
        assert abs(association['material_gradient_projection_slope']-1) < 1e-5
        count += 1
    assert count == 24
    return dict(status='CPU_SELFCHECK_PASS', count=count, inference_executed=False,
                checks=['truth decomposition','normal/direction unit lengths','identity relighting',
                        'shading scale invariance','deliberate material leakage detection'])


def infer(output, old_root):
    assets = verify_assets(old_root)
    manifest = generate(output)
    (output/'predictions').mkdir(exist_ok=True)
    (output/'metrics').mkdir(exist_ok=True)
    import torch
    import cv2
    if not torch.cuda.is_available():
        raise RuntimeError('Inherited frozen-prior loaders require existing CUDA runtime')
    sys.path.insert(0, str(old_root))
    spec = importlib.util.spec_from_file_location('inherited_prior_loader', old_root/'qc_code_002.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    gray, normal_model, run_gray = adapter.load_priors()
    from chrislib.general import uninvert
    decoder_probe = np.array([.25,1,4,20],np.float32)
    np.testing.assert_allclose(uninvert(1/(1+decoder_probe)),decoder_probe,rtol=1e-5)
    # Track all identities before inference; no optimizer or training is created.
    import importlib.metadata as md
    runtime = dict(status='IN_PROGRESS', assets=assets, code_sha256=sha(__file__),
                   truth_manifest_sha256=sha(output/'truth_manifest.json'),
                   device=torch.cuda.get_device_name(), versions={k:md.version(k) for k in
                   ['torch','torchvision','numpy','geffnet','timm','scipy','scikit-image','kornia']})
    save_json(output/'runtime.json', runtime)
    rows = []
    start = time.time()
    for entry in manifest['observations']:
        with np.load(output/entry['file'],allow_pickle=False) as f:
            truth = {k:f[k] for k in f.files}
        w, support = truth['input_linear'], truth['support']
        proxy_scale = float(w.max())
        proxy = w/proxy_scale
        with torch.inference_mode():
            decomposition = run_gray(gray,proxy,linear=True,resize_conf=1024,maintain_size=False,device='cuda')
        decoded = np.asarray(uninvert(decomposition['gry_shd']),np.float32)
        if decoded.ndim != 2:
            raise RuntimeError('Unexpected inherited shading shape')
        good = np.isfinite(decoded) & (decoded>0)
        reconstruction = decomposition['gry_alb']*decoded[...,None]
        original = decomposition['lin_img']
        reconstruction_valid = good[...,None]&np.isfinite(reconstruction)&np.isfinite(original)
        if not reconstruction_valid.any():
            raise RuntimeError('Empty inherited decomposition reconstruction support')
        reconstruction_max_abs = float(np.max(np.abs(reconstruction-original)[reconstruction_valid]))
        if reconstruction_max_abs > 1e-5:
            raise RuntimeError('Inherited inverse-shading unit check failed')
        raw = cv2.resize(np.where(good,decoded,0),(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
        shading_valid = (cv2.resize(good.astype(np.float32),(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)>=1)&np.isfinite(raw)&(raw>0)
        median_support = shading_valid & support
        if not median_support.any():
            raise RuntimeError('Empty shading median support')
        relative = raw/float(np.median(raw[median_support]))
        s0 = np.where(shading_valid,np.clip(relative,.25,4),1).astype(np.float32)
        n, normal_valid = adapter.normal_inference(normal_model,proxy)
        if n.shape != truth['normal'].shape:
            raise RuntimeError('Unexpected inherited normal shape')
        # Both raw decoded shading and actual clipped production factor are scored.
        safe_raw = np.where(shading_valid,raw,1).astype(np.float32)
        metrics = evaluate(truth,safe_raw,n,shading_valid,normal_valid)
        production = evaluate(truth,s0,n,shading_valid,normal_valid)
        metrics['production_s0'] = production
        metrics['shading_lower_clip_fraction'] = float((relative[median_support]<.25).mean())
        metrics['shading_upper_clip_fraction'] = float((relative[median_support]>4).mean())
        metrics['proxy_scale'] = proxy_scale
        metrics['algebraic_reconstruction_max_abs'] = reconstruction_max_abs
        prediction_path = output/'predictions'/f"{entry['id']}.npz"
        np.savez_compressed(prediction_path,shading_raw=raw,s0=s0,normal=n,
                            shading_valid=shading_valid,normal_valid=normal_valid)
        row = dict(id=entry['id'],truth_sha256=entry['sha256'],
                   prediction_sha256=sha(prediction_path),metrics=metrics)
        save_json(output/'metrics'/f"{entry['id']}.json",row)
        rows.append(row)
        save_json(output/'summary.json',dict(status='IN_PROGRESS',count=len(rows),observations=rows,limitations=LIMITATIONS))
        print(json.dumps(dict(id=entry['id'],completed=len(rows),normal_mean_deg=metrics['normal_mean_deg'])),flush=True)
    # Descriptive mean over observations; not an independent-scene confidence interval.
    aggregate = {key:float(np.mean([r['metrics'][key] for r in rows])) for key in ['normal_mean_deg','normal_median_deg','normal_p95_deg']}
    aggregate['shading_log_rmse_observation_mean'] = float(np.mean([r['metrics']['shading_raw']['log_rmse'] for r in rows]))
    aggregate['production_s0_log_rmse_observation_mean'] = float(np.mean([r['metrics']['production_s0']['shading_raw']['log_rmse'] for r in rows]))
    with (output/'per_observation.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['id','normal_mean_deg','shading_log_rmse','production_s0_log_rmse','material_gradient_correlation','common_pixels'])
        writer.writeheader()
        for r in rows:
            m = r['metrics']
            writer.writerow(dict(id=r['id'],normal_mean_deg=m['normal_mean_deg'],shading_log_rmse=m['shading_raw']['log_rmse'],
                                 production_s0_log_rmse=m['production_s0']['shading_raw']['log_rmse'],
                                 material_gradient_correlation=m['material_detail']['adjacent_log_gradient_correlation'],common_pixels=m['common_pixels']))
    runtime.update(status='INFERENCE_COMPLETE_REVIEW_PENDING',seconds=time.time()-start,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save_json(output/'runtime.json',runtime)
    save_json(output/'summary.json',dict(status='INFERENCE_COMPLETE_REVIEW_PENDING',count=len(rows),
                                        descriptive_observation_means=aggregate,observations=rows,limitations=LIMITATIONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['selfcheck','generate','assets','infer'],required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'prior_calibration')
    parser.add_argument('--old-root',type=Path,help='Existing private diffuse_vgt_v1 runtime with vendor/weights')
    args = parser.parse_args()
    if args.mode == 'selfcheck':
        print(json.dumps(selfcheck(),indent=2))
        return
    if args.mode == 'generate':
        manifest = generate(args.output)
        print(json.dumps(dict(status='TRUTHS_GENERATED',count=len(manifest['observations']),inference_executed=False)))
        return
    if args.old_root is None:
        parser.error('--old-root is required for assets/infer')
    if args.mode == 'assets':
        print(json.dumps(verify_assets(args.old_root),indent=2))
        return
    try:
        infer(args.output,args.old_root)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        save_json(args.output/'failure.json',dict(status='FAILED',type=type(exc).__name__,message=str(exc)))
        raise


if __name__ == '__main__':
    main()

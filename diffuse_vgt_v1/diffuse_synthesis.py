"""Revision-1 Torch adapter. Implementation candidate, not training clearance.

Cache tensors are supplied by the caller: no dataset/GT reads occur here.
The parent capture adapter preserves its RNG calls and original-image branch.
"""
import math
import hashlib
import torch
import torch.nn.functional as F


def require_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'Nonfinite {name}')


def sample_lamps(n, basis, generator):
    """basis rows = orthonormal tangent-x, tangent-y, verified front direction."""
    if basis.shape != (3, 3):
        raise ValueError('Expected explicit 3x3 normal-coordinate basis')
    require_finite(basis, 'basis')
    if not torch.allclose(basis @ basis.T, torch.eye(3, device=basis.device, dtype=basis.dtype), atol=1e-5, rtol=0):
        raise ValueError('Basis must be orthonormal')
    # Dedicated generator; these draws never advance the parent streams.
    cos_theta = math.cos(math.radians(70)) + (1 - math.cos(math.radians(70))) * torch.rand(n, 2, device=basis.device, dtype=basis.dtype, generator=generator)
    phi = 2 * math.pi * torch.rand(n, 2, device=basis.device, dtype=basis.dtype, generator=generator)
    sin_theta = (1 - cos_theta.square()).clamp_min(0).sqrt()
    directions = torch.stack((sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta), -1) @ basis
    stops = 2 * torch.rand(n, device=basis.device, dtype=basis.dtype, generator=generator) - 1
    powers = torch.stack((2 ** (stops / 2), 2 ** (-stops / 2)), -1)
    return directions, powers, dict(cos_theta=cos_theta, phi=phi, relative_stops=stops)


def replay_crop_plan(n, device, generator):
    """Exactly round2.crop_views draws, with an independent original-state gc."""
    chosen = torch.randperm(n, generator=generator, device=device)[:n // 2]
    sizes = torch.randint(384, 513, (len(chosen),), generator=generator, device=device)
    left = (torch.rand(len(chosen), generator=generator, device=device) * (513 - sizes)).long()
    top = (torch.rand(len(chosen), generator=generator, device=device) * (513 - sizes)).long()
    return dict(chosen=chosen, sizes=sizes, left=left, top=top)


@torch.no_grad()
def resample_priors(cache, crop_plan, crop_grid, facing_normal, chunk_size=16):
    """Cache: s0, normal and separate shading_valid/normal_valid bool masks.

    validity is transported as invalid contribution mass: accept iff zero.
    bilinear border padding and 2x2 averaging exactly match parent's crop.
    Full-image views use 2x2 averaging, matching 512->256 INTER_AREA.
    """
    s, normal = cache['s0'], cache['normal']
    shading_valid, normal_valid = cache['shading_valid'], cache['normal_valid']
    if s.ndim != 4 or s.shape[1:] != (1, 512, 512) or normal.shape != (len(s), 3, 512, 512) or any(v.shape != s.shape or v.dtype != torch.bool for v in (shading_valid, normal_valid)):
        raise ValueError('Expected native 512 cache with bool validity')
    normal = normal.float()
    s = s.float()
    front = facing_normal.to(device=s.device, dtype=s.dtype)
    require_finite(front, 'facing normal')
    if front.shape != (3,) or not torch.allclose(front.norm(), front.new_tensor(1), atol=1e-6, rtol=0):
        raise ValueError('Facing normal must be unit')
    norm = normal.norm(dim=1, keepdim=True)
    nv = normal_valid & torch.isfinite(normal).all(1, keepdim=True) & torch.isfinite(norm) & (norm > 0)
    sv = shading_valid & torch.isfinite(s) & (s > 0)
    if (sv & ((s < .25) | (s > 4))).any():
        raise ValueError('Cache s0 outside fixed bounds')
    normal = torch.where(nv, normal / norm.clamp_min(1e-20), front[None, :, None, None])
    s = torch.where(sv, s, torch.ones_like(s))
    # Keep normal/shading validity separate: C rotates only normal-derived H.
    packed = torch.cat((s, normal, (~sv).float(), (~nv).float()), 1)
    aligned = F.avg_pool2d(packed, 2).repeat_interleave(2, 0)
    chosen = crop_plan['chosen']
    for start in range(0, len(chosen), chunk_size):
        sl = slice(start, start + chunk_size)
        idx = chosen[sl]
        grid = crop_grid(crop_plan['sizes'][sl], crop_plan['left'][sl], crop_plan['top'][sl])
        sampled = F.grid_sample(packed[idx // 2], grid, mode='bilinear', padding_mode='border', align_corners=False)
        aligned[idx] = F.avg_pool2d(sampled, 2)
    s, normal = aligned[:, :1], aligned[:, 1:4]
    norms = normal.norm(dim=1, keepdim=True)
    nv = (aligned[:, 5:6] == 0) & torch.isfinite(normal).all(1, keepdim=True) & torch.isfinite(norms) & (norms > 1e-12)
    sv = (aligned[:, 4:5] == 0) & torch.isfinite(s) & (s > 0)
    normal = torch.where(nv, normal / norms.clamp_min(1e-20), front[None, :, None, None])
    s = torch.where(sv, s, torch.ones_like(s))
    common = sv & nv & torch.flip(nv, (-2, -1))
    return dict(s0=s, normal=normal, shading_valid=sv, normal_valid=nv,
                valid=common, fill_normal=front,
                invalid_fraction=(~common).float().mean((1, 2, 3)))


@torch.no_grad()
def render_fields(wb, priors, endpoints, directions, powers, *, eta=.1):
    """NCHW W already in inherited /4 domain. Return all counterfactuals.

    Median is arithmetic middle-pair median (quantile .5), matching NumPy.
    Targets J remain float32. No clipping or a second /4 is applied.
    """
    wb = wb.float()
    require_finite(wb, 'W')
    if (wb < 0).any():
        raise ValueError('Negative W')
    for value, name in ((endpoints, 'endpoints'), (directions, 'directions'), (powers, 'powers'), (priors['s0'], 's0'), (priors['normal'], 'normal')):
        require_finite(value, name)
    if not 0 < eta <= 1 or (endpoints <= 0).any() or not torch.equal(endpoints[:, :, 1], torch.ones_like(endpoints[:, :, 1])) or (powers <= 0).any():
        raise ValueError('Invalid lamp parameters')
    if not torch.allclose(directions.norm(dim=-1), torch.ones_like(powers), atol=1e-5, rtol=0):
        raise ValueError('Lamp directions must be unit')
    if (priors['s0'] <= 0).any():
        raise ValueError('Unfilled nonpositive shading')
    u = powers[:, :, None, None] * (eta + (1 - eta) * torch.einsum('nchw,nkc->nkhw', priors['normal'], directions).clamp_min(0))
    Hraw = torch.einsum('nkhw,nkc->nchw', u, endpoints)
    Traw = u.sum(1, keepdim=True)
    require_finite(Hraw, 'raw H')
    median = torch.quantile(Traw.flatten(1), .5, dim=1)[:, None, None, None]
    if (median <= 0).any():
        raise FloatingPointError('Zero median')
    H = Hraw / median
    texture = wb / priors['s0']
    arms = {}
    for name, base, field in (('A', wb, H), ('B', texture, H), ('C', texture, torch.flip(H, (-2, -1)))):
        T = field[:, 1:2]
        arms[name] = dict(H=field, T=T, E=field / T, I=base * field, J=base * T)
    peak = torch.stack([arm['I'].flatten(1).amax(1) for arm in arms.values()]).amax(0)
    require_finite(peak, 'counterfactual peak')
    factor = torch.pow(2., -torch.ceil(torch.log2((peak / 60000).clamp_min(1))))
    if not torch.isfinite(factor).all() or (factor <= 0).any():
        raise FloatingPointError('Invalid common exposure')
    for arm in arms.values():
        arm['I'] = arm['I'] * factor[:, None, None, None]
        arm['J'] = arm['J'] * factor[:, None, None, None]
        for value in arm.values():
            require_finite(value, 'rendered output')
        if arm['I'].max() > 60000:
            raise FloatingPointError('Exposure bound exceeded')
    return dict(arms=arms, exposure=factor, peak=peak, H_raw=Hraw,
                T_raw=Traw, alpha=u[:, :1] / Traw, u=u, median=median)


def patch_mean(field):
    n, c, h, w = field.shape
    if (h, w) != (256, 256):
        raise ValueError('Expected frozen 256 grid')
    return field.reshape(n, c, 16, 16, 16, 16).mean((3, 5)).permute(0, 2, 3, 1).reshape(n, 256, c)


@torch.no_grad()
def adapt_capture(capture, priors, endpoints, directions, powers, base, *, mode, chunk_size=16):
    """Return parent-compatible five tensors plus diagnostics, before ledger.

    Parent capture must come from existing dg_synthesis mode='O'. No RNG here.
    All three arm half-storage masks are checked against the common source mask;
    lamp-induced rejection raises, rather than changing the consumption mask.
    """
    if mode not in ('A', 'B', 'C'):
        raise ValueError('Expected A, B, or C')
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    wb, branch, patches, order = (capture[k] for k in ('wb', 'branch', 'patches', 'order'))
    if not torch.all((branch == 0) | (branch == 2)):
        raise ValueError('Only frozen O branch composition supported')
    n = len(wb)
    parent_nonblack = wb.reshape(n, 3, 16, 16, 16, 16).amax((1, 3, 5)).reshape(n, 256) > 0
    technical = priors['valid'].reshape(n, 1, 16, 16, 16, 16).all(1).all(2).all(3).reshape(n, 256)
    common = parent_nonblack & technical
    common = torch.gather(common, 1, patches)
    # Identity uses the exact parent's float image and label construction.
    identity_rgb = capture['frameFloat']
    identity_E = capture['light'].permute(0, 3, 1, 2)
    # Only selected half frames and patch GT survive each chunk. All three
    # float counterfactuals remain bounded by chunk_size, never cohort size.
    frame = torch.empty_like(wb, dtype=torch.float16)
    gt = torch.empty((n, 256, 3), dtype=torch.float32, device=wb.device)
    exposure_min = 1.
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        count = sl.stop - start
        subpriors = {key: priors[key][sl] for key in ('s0', 'normal')}
        fields = render_fields(wb[sl], subpriors, endpoints[sl], directions[sl], powers[sl])
        mixed = (branch[sl] == 2)[:, None, None, None]
        if mixed.any():
            exposure_min = min(exposure_min, float(fields['exposure'][branch[sl] == 2].min()))
        for name, arm in fields['arms'].items():
            rgb = torch.where(mixed, arm['I'], identity_rgb[sl])
            E = torch.where(mixed, arm['E'], identity_E[sl])
            chunk_frame = rgb.half()
            require_finite(chunk_frame, 'half frame')
            chunk_gt = patch_mean(E)
            normalized_gt = chunk_gt / chunk_gt.norm(dim=2, keepdim=True).clamp_min(base.EPS)
            actual = chunk_frame.reshape(count, 3, 16, 16, 16, 16).amax((1, 3, 5)).reshape(count, 256) > 0
            actual &= normalized_gt.amin(2) > 1e-4
            actual = torch.gather(actual, 1, patches[sl])
            if not torch.equal(actual & common[sl], common[sl]):
                raise FloatingPointError('Lighting/half storage invalidated common accepted patch')
            if name == mode:
                frame[sl] = chunk_frame
                gt[sl] = chunk_gt
        del fields, arm, rgb, E, chunk_frame, chunk_gt, normalized_gt, actual
    ordered_valid = common[order].flatten()
    if not ordered_valid.reshape(-1, 512).any(1).all():
        raise FloatingPointError('Empty common batch: cannot advance optimizer budget')
    geom = capture['geom']
    imgs = base.rot_flip(frame, *geom)
    pp = base.extract_patches(frame)
    pp = base.rot_flip(pp.reshape(n * 256, 3, 16, 16), *(x.repeat_interleave(256) for x in geom)).reshape(n, 256, 3, 16, 16)
    pp = torch.gather(pp, 1, patches[..., None, None, None].expand(-1, -1, 3, 16, 16))
    gt = torch.gather(gt, 1, patches[..., None].expand(-1, -1, 3))[order].reshape(-1, 3)
    image_z = base.zscore_bc(imgs[order].float())
    require_finite(image_z, 'zscore image')
    sampled_parent = torch.gather(parent_nonblack, 1, patches)
    used = capture['consumed']
    diag = dict(mode=mode, valid_patches=int(common.sum()), parent_valid_patches=int(sampled_parent.sum()),
                technical_rejected_patches=int((sampled_parent & ~common).sum()),
                consumed_valid_patches=int(common[used].sum()),
                common_arm_mask_verified=True, exposure_min=exposure_min,
                branch_counts=torch.bincount(branch, minlength=3).tolist(),
                invalid_fraction=priors['invalid_fraction'].cpu().tolist())
    return (image_z, pp[order].reshape(-1, 3, 16, 16), gt / gt.norm(dim=1, keepdim=True).clamp_min(base.EPS),
            ordered_valid, torch.arange(n, device=wb.device).repeat_interleave(64), diag), common


def select_consumed_batches(order, batch_indices):
    """Map explicit optimizer batch indices back to unshuffled view slots."""
    indices = list(batch_indices)
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or (i + 1) * 8 > len(order) for i in indices):
        raise ValueError('Invalid actual optimizer batch indices')
    consumed = torch.zeros(len(order), device=order.device, dtype=torch.bool)
    for index in indices:
        consumed[order[index * 8:(index + 1) * 8]] = True
    return consumed


def consumption_ledger(parent_capture, common, pool_size):
    mixed_used = parent_capture['consumed'] & (parent_capture['branch'] == 2)
    patch_mass = (common.sum(1) * mixed_used).reshape(-1, 2).sum(1)
    view_mass = mixed_used.reshape(-1, 2).sum(1)
    ledger = {}
    for name, index, mass in [('first_patch_counts', parent_capture['first'], patch_mass), ('second_patch_counts', parent_capture['second'], patch_mass), ('first_view_counts', parent_capture['first'], view_mass), ('second_view_counts', parent_capture['second'], view_mass)]:
        ledger[name] = torch.zeros(pool_size, dtype=torch.long, device=common.device).scatter_add_(0, index, mass).cpu().numpy()
    return ledger, int(patch_mass.sum()), int(view_mass.sum())


@torch.no_grad()
def render_cycle(source, pool, bank, config, cycle, seed, remaining_steps, *, cache, basis, mode, capture=None, batch_indices=None):
    """Integration candidate: reuse frozen parent execution as draw/crop capture.

    Caller must verify parent hashes and cache scene order before use. This
    convenience route runs old O once to capture exact draws; optimize only after
    equivalence testing. No training is started by this function.
    """
    import dg_synthesis as dg
    if config != dg.parent.CONFIGS['O']:
        raise ValueError('Only frozen O config supported')
    parent_capture = {}
    dg.render_cycle(source, pool, bank, config, cycle, seed, remaining_steps, mode='O', capture=parent_capture)
    if batch_indices is not None:
        if any(i >= remaining_steps for i in batch_indices):
            raise ValueError('Batch outside requested cycle prefix')
        parent_capture['consumed'] = select_consumed_batches(parent_capture['order'], batch_indices)
    device = source[1].device
    n = len(source[1]) * 2
    gc = dg.v1.generator(device, seed * 1000003 + cycle * 13 + 6)
    crop_plan = replay_crop_plan(n, device, gc)
    priors = resample_priors(cache, crop_plan, dg.r2.crop_grid, basis[2])
    # Separate large offset and own generator; parent gg remains untouched.
    lamp_gen = dg.v1.generator(device, seed * 1000003 + cycle * 1009 + 700000033)
    directions, powers, parameters = sample_lamps(n, basis, lamp_gen)
    first, second = parent_capture['first'], parent_capture['second']
    endpoints = torch.stack((pool[first], pool[second]), 1).repeat_interleave(2, 0)
    result, common = adapt_capture(parent_capture, priors, endpoints, directions, powers, dg.base, mode=mode)
    ledger, patch_mass, view_mass = consumption_ledger(parent_capture, common, len(pool))
    result[-1].update(consumed_mixed_valid_patches=patch_mass, consumed_mixed_views=view_mass,
                      consumed_views=int(parent_capture['consumed'].sum()),
                      consumed_batch_indices=list(batch_indices) if batch_indices is not None else list(range(min(n // 8, remaining_steps))))
    sampling_hash = hashlib.sha256()
    for name, value in [(key, parent_capture[key]) for key in ('first', 'second', 'patches', 'branch', 'order', 'consumed')] + [(f'geom{i}', x) for i, x in enumerate(parent_capture['geom'])] + list(crop_plan.items()) + [('common_valid', common), ('directions', directions), ('powers', powers)]:
        sampling_hash.update(name.encode('ascii'))
        sampling_hash.update(str(tuple(value.shape)).encode('ascii'))
        sampling_hash.update(value.detach().contiguous().cpu().numpy().tobytes())
    result[-1]['matched_sampling_sha256'] = sampling_hash.hexdigest()
    if capture is not None:
        capture.update(parent=parent_capture, crop_plan=crop_plan, directions=directions, powers=powers, lamp_parameters=parameters, common_valid=common)
    return (*result, ledger)

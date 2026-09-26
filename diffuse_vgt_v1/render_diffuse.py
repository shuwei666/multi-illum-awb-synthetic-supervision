"""Pure NumPy diffuse-VGT revision-1 renderer; no data or model loading.

All vectors share an explicitly supplied coordinate basis. Geometry is already
aligned to the training grid. This module makes no DSINE coordinate claim.
"""
import numpy as np


def _finite(x, name):
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")


def render_diffuse(W, s0, normals, technical_valid, lights, directions, powers,
                   *, facing_normal, eta=0.1, storage_divisor=4.0,
                   storage_target=60000.0):
    """Return fields and arms A/B/C, sharing technical mask and exposure.

    W: HxWx3 floating, nonnegative camera-linear image. s0: HxW bounded
    relative shading. normals: HxWx3 unit vectors on valid support.
    technical_valid: HxW boolean (must include conservative alignment validity).
    lights/directions: 2x3; lights G=1, directions unit length. powers: 2.
    facing_normal: explicit verified fill vector, never grants validity.
    Exposure is a power of two <=1 protecting I/storage_divisor and J likewise.
    Nonfinite/zero priors are filled and rejected; nonfinite radiometry fails.
    """
    W = np.asarray(W)
    if W.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('W must be float32 or float64')
    dtype = W.dtype
    s = np.asarray(s0, dtype=dtype)
    n = np.asarray(normals, dtype=dtype)
    valid = np.asarray(technical_valid)
    if W.ndim != 3 or W.shape[-1] != 3 or 0 in W.shape:
        raise ValueError('W must be nonempty HxWx3')
    if s.shape != W.shape[:2] or n.shape != W.shape or valid.shape != s.shape or valid.dtype != bool:
        raise ValueError('prior shapes or validity dtype mismatch')
    _finite(W, 'W')
    if np.any(W < 0):
        raise ValueError('W must be nonnegative')
    L = np.asarray(lights, dtype=dtype)
    d = np.asarray(directions, dtype=dtype)
    a = np.asarray(powers, dtype=dtype)
    front = np.asarray(facing_normal, dtype=dtype)
    for x, shape, name in ((L, (2, 3), 'lights'), (d, (2, 3), 'directions'),
                            (a, (2,), 'powers'), (front, (3,), 'facing_normal')):
        if x.shape != shape:
            raise ValueError(f'{name} shape mismatch')
        _finite(x, name)
    if np.any(L <= 0) or not np.all(L[:, 1] == 1) or np.any(a <= 0):
        raise ValueError('positive powers and positive G=1 lights required')
    if not np.allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-6, rtol=0) or not np.isclose(np.linalg.norm(front), 1, atol=1e-6, rtol=0):
        raise ValueError('directions and facing_normal must be unit vectors')
    if not (np.isfinite(eta) and 0 < eta <= 1 and np.isfinite(storage_divisor) and storage_divisor > 0 and np.isfinite(storage_target) and storage_target > 0):
        raise ValueError('invalid eta or storage bounds')
    with np.errstate(over='ignore', invalid='ignore'):
        norms = np.linalg.norm(n, axis=-1)
    normal_valid = np.all(np.isfinite(n), axis=-1) & np.isfinite(norms) & (norms > 0)
    shading_valid = np.isfinite(s) & (s > 0)
    local_valid = valid & normal_valid & shading_valid
    if np.any(local_valid & ((s < .25) | (s > 4))):
        raise ValueError('valid s0 must already be clipped to [0.25, 4]')
    if np.any(valid & normal_valid & (np.abs(norms - 1) > 1e-5)):
        raise ValueError('valid normals must be unit vectors')
    # A caller's technical-invalid location may encode any untrusted prior.
    n = np.where((valid & normal_valid)[..., None], n, front)
    s = np.where(local_valid, s, dtype.type(1))
    common_valid = local_valid & np.rot90(valid & normal_valid, 2)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        u = a * (dtype.type(eta) + dtype.type(1 - eta) * np.maximum(dtype.type(0), n @ d.T))
        H_raw = u @ L
        T_raw = u.sum(axis=-1)
        median = np.median(T_raw)
        if not np.isfinite(median) or median <= 0:
            raise ValueError('invalid median intensity')
        H = H_raw / median
        alpha = u[..., 0] / T_raw
        base = W / s[..., None]
        arms = {}
        for name, texture, field in (('A', W, H), ('B', base, H), ('C', base, np.rot90(H, 2))):
            T = field[..., 1]
            E = field / T[..., None]
            arms[name] = dict(H=field, T=T, E=E, I=texture * field, J=texture * T[..., None])
        maximum = max(float(np.max(arm[key])) for arm in arms.values() for key in ('I', 'J'))
        limit = float(storage_target) * float(storage_divisor)
        if not np.isfinite(limit):
            raise ValueError('storage bound overflow')
        exponent = min(0, int(np.floor(np.log2(limit) - np.log2(maximum)))) if maximum > 0 else 0
        exposure = dtype.type(np.ldexp(1.0, exponent))
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError('exposure underflow')
        for arm in arms.values():
            arm['I'] = arm['I'] * exposure
            arm['J'] = arm['J'] * exposure
            for value in arm.values():
                _finite(value, 'rendered field')
        # Guard rounding at the storage boundary without clipping any pixel.
        if any(np.max(arm[k] / storage_divisor) > storage_target for arm in arms.values() for k in ('I', 'J')):
            exposure *= dtype.type(.5)
            for arm in arms.values():
                arm['I'] *= dtype.type(.5)
                arm['J'] *= dtype.type(.5)
    return dict(arms=arms, H_raw=H_raw, T_raw=T_raw, u=u, alpha=alpha,
                median=median, exposure=exposure, technical_valid=common_valid,
                local_valid=local_valid, normal_valid=valid & normal_valid,
                fill_normal=front.copy(), filled_s0=s,
                invalid_fraction=float(1 - common_valid.mean()))


def patch_labels(rendered, patches, parent_patch_valid=None, patch_size=16):
    """Unmasked patch mean of E then L2; accept only all-technical-valid.

    patches is Nx2 (top,left); parent_patch_valid preserves parent's nonblack
    or other patch rejection. Invalid labels are finite but MUST NOT be used.
    """
    positions = np.asarray(patches)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError('patches must be Nx2 integer coordinates')
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError('patch_size must be positive integer')
    mask = rendered['technical_valid']
    accepted = np.ones(len(positions), dtype=bool) if parent_patch_valid is None else np.asarray(parent_patch_valid).copy()
    if accepted.dtype != bool or accepted.shape != (len(positions),):
        raise ValueError('parent_patch_valid must be N boolean values')
    labels = {name: [] for name in rendered['arms']}
    for i, (top, left) in enumerate(positions):
        if top < 0 or left < 0 or top + patch_size > mask.shape[0] or left + patch_size > mask.shape[1]:
            raise ValueError('patch outside image')
        sl = np.s_[top:top + patch_size, left:left + patch_size]
        accepted[i] &= np.all(mask[sl])
        for name, arm in rendered['arms'].items():
            mean = arm['E'][sl].mean(axis=(0, 1))
            _finite(mean, 'patch mean')
            scale = np.max(np.abs(mean))
            if scale <= 0:
                raise ValueError('zero patch label')
            scaled = mean / scale
            labels[name].append(scaled / np.linalg.norm(scaled))
    dtype = rendered['arms']['A']['E'].dtype
    return {name: np.asarray(values, dtype=dtype).reshape(-1, 3) for name, values in labels.items()}, accepted

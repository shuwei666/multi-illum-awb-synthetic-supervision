"""Seven-arm adapter; no random draws, no mutation of the frozen producer."""
ARMS = ('ES00', 'ES11', 'ES10', 'ES01', 'KEEP_OLD_R', 'SHAM_OLD_0', 'NO_NEW_S')

def fields(producer, wb, priors, endpoints, directions, powers, **kwargs):
    import torch
    original = producer(wb, priors, endpoints, directions, powers, **kwargs)
    H = original['arms']['A']['H']
    S = H[:, 1:2]
    E = H / S
    q = lambda x: torch.flip(x, (-2, -1))
    W = wb.float()
    R = W / priors['s0']
    definitions = {
        'ES00': (R * H, E, R * S),
        'ES11': (R * q(H), q(E), R * q(S)),
        'ES10': (R * S * q(E), q(E), R * S),
        'ES01': (R * q(S) * E, E, R * q(S)),
        'KEEP_OLD_R': (W * q(H), q(E), W * q(S)),
        'SHAM_OLD_0': ((W / q(priors['s0'])) * H, E, (W / q(priors['s0'])) * S),
        'NO_NEW_S': (R * E, E, R),
    }
    peak = torch.stack([x[0].flatten(1).amax(1) for x in definitions.values()] + [(W * H).flatten(1).amax(1)]).amax(0)
    factor = torch.pow(2., -torch.ceil(torch.log2((peak / 60000).clamp_min(1))))
    result = {}
    for name, (I, label, J) in definitions.items():
        I, J = I * factor[:, None, None, None], J * factor[:, None, None, None]
        if not all(bool(torch.isfinite(x).all()) for x in (I, label, J)) or not bool((label > 0).all()) or I.max() > 60000:
            raise FloatingPointError('Invalid seven-arm render: ' + name)
        if not torch.allclose(I / label, J, atol=.02, rtol=3e-6):
            raise FloatingPointError('I/E=J mismatch: ' + name)
        result[name] = dict(I=I, E=label, J=J)
    return dict(original, arms=result, exposure=factor, peak=peak)

class Adapter:
    def __init__(self, synth):
        self.synth = synth
        self.original_fields = synth.render_fields
        self.original_resample = synth.resample_priors
        self.selected = None
        synth.render_fields = self.render_fields
        synth.resample_priors = self.resample

    def resample(self, *args, **kwargs):
        import torch
        p = self.original_resample(*args, **kwargs)
        p['valid'] = p['valid'] & torch.flip(p['shading_valid'], (-2, -1))
        p['invalid_fraction'] = (~p['valid']).float().mean((1, 2, 3))
        return p

    def render_fields(self, *args, **kwargs):
        out = fields(self.original_fields, *args, **kwargs)
        # Parent adapt_capture validates every counterfactual then stores B.
        out['arms']['B'] = out['arms'][self.selected]
        return out

    def render_cycle(self, *args, mode, **kwargs):
        if mode not in ARMS:
            raise ValueError('Unknown seven arm')
        self.selected = mode
        result = self.synth.render_cycle(*args, mode='B', **kwargs)
        result[5]['seven_arm'] = mode
        return result

    def __getattr__(self, name):
        return getattr(self.synth, name)

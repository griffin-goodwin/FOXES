"""Translate the three independent model sections into constructor settings."""
from copy import deepcopy


def resolve_model_settings(config):
    config = deepcopy(config)
    if config.get('model_type', 'deterministic') != 'uncertainty':
        return config
    removed = {'mean_parameterization', 'contrast_prior', 'finetune'} & config.keys()
    if removed:
        raise ValueError(f'Removed settings: {sorted(removed)}; use the simplified training config')
    architecture = config.setdefault('vit_architecture', {})
    uncertainty = config.setdefault('uncertainty', {})
    if 'global_head' in config:
        if 'patch_global_context' in architecture:
            raise ValueError('Use global_head, not both global_head and vit_architecture.patch_global_context')
        architecture['patch_global_context'] = dict(config['global_head'])
    mean_loss = config.get('mean_loss', {})
    if set(mean_loss) - {'huber_delta', 'class_weighting', 'class_weight_exponent'}:
        raise ValueError('Unknown mean_loss settings')
    for key, value in mean_loss.items():
        if key in uncertainty:
            raise ValueError(f'Configure {key} only in mean_loss')
        uncertainty[key] = value
    local_head = config.get('local_head', {})
    if set(local_head) - {'max_abs_log10_multiplier'}:
        raise ValueError('Unknown local_head settings')
    if 'max_abs_log10_multiplier' in local_head:
        uncertainty['max_abs_log10_patch_multiplier'] = local_head['max_abs_log10_multiplier']
    return config

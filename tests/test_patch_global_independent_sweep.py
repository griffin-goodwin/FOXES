from copy import deepcopy
import pytest
import yaml
from experiments.patch_global_independent_sweep.run_sweep import ARMS, HERE, validate_configs


def test_sweep_factors_and_reference():
    configs = validate_configs()
    assert len(configs) == 3
    ref = configs['range1p5_huber0p15']
    assert ref['uncertainty']['enabled']
    assert all(c['checkpoint']['initialize_from'] is None and c['checkpoint']['resume_from'] is None
               for c in configs.values())
    assert all(c['vit_architecture']['patch_global_context']['max_abs_adjustment'] == 1.5
               for c in configs.values())
    assert {c['uncertainty']['huber_delta'] for c in configs.values()} == {0.15, 0.3, 1.0}


@pytest.mark.parametrize('section,key,value,match', [
    ('checkpoint', 'initialize_from', '/tmp/old.ckpt', 'from scratch'),
    ('vit_architecture', 'dropout', 0.5, 'outside Huber'),
    ('optimizer', 'weight_decay', 0.5, 'outside Huber'),
    ('data', 'aia_dir', '/data/FOXES_OG/AIA_processed', 'screening subset'),
])
def test_rejects_uncontrolled_changes(tmp_path, section, key, value, match):
    configs = deepcopy(validate_configs())
    configs[next(iter(ARMS))][section][key] = value
    for name, config in configs.items():
        for key in ('global_head', 'mean_loss', 'local_head'):
            config.pop(key, None)
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        validate_configs(tmp_path)


def test_matched_branch_toggles_are_supported(tmp_path):
    for name in ARMS:
        config = yaml.safe_load((HERE/f'{name}.yaml').read_text())
        config['global_head']['enabled'] = False
        config['uncertainty']['enabled'] = False
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(config))
    configs = validate_configs(tmp_path)
    assert all(not c['vit_architecture']['patch_global_context']['enabled']
               and not c['uncertainty']['enabled'] for c in configs.values())

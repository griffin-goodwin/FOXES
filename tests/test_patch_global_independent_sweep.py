from copy import deepcopy
import pytest
import yaml
from experiments.patch_global_independent_sweep.run_sweep import ARMS, validate_configs


def test_sweep_factors_and_reference():
    configs = validate_configs()
    assert len(configs) == 6
    ref = configs['range1p5_huber0p15']
    assert ref['vit_architecture']['independent_local_global']
    assert ref['uncertainty']['enabled']
    assert {c['uncertainty']['huber_delta'] for c in configs.values()} == {0.15, 0.3}


@pytest.mark.parametrize('section,key,value,match', [
    ('vit_architecture', 'independent_local_global', False, 'independent'),
    ('optimizer', 'weight_decay', 0.5, 'outside scaling'),
    ('data', 'aia_dir', '/data/FOXES_OG/AIA_processed', 'screening subset'),
])
def test_rejects_uncontrolled_changes(tmp_path, section, key, value, match):
    configs = deepcopy(validate_configs())
    configs[next(iter(ARMS))][section][key] = value
    for name, config in configs.items():
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        validate_configs(tmp_path)

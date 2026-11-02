"""Workflow paths must not change when launched from another directory."""

from pathlib import Path

import pytest
import yaml

from forecasting.config import load_config


@pytest.mark.parametrize('path_root', [None, '..'])
def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch, path_root):
    folder = tmp_path / 'project' / 'configs'
    folder.mkdir(parents=True)
    config_path = folder / 'inference.yaml'
    absolute = str(tmp_path / 'external' / 'norm.npz')
    config = {
        'data': {'aia_dir': './images', 'checkpoint_path': 'model.ckpt',
                 'sxr_norm_path': absolute, 'aia_norm_path': None},
        'output_path': 'outputs/predictions.csv',
        'weight_path': '',
        'checkpoint': {'monitor': 'val/total_loss'},
        'repo_id': 'griffingoodwin04/FOXES',
        'plotting': {'inference_patch_dir': '${FOXES_TEST_DATA}/maps'},
        'save_dir': '~/foxes-downloads',
    }
    if path_root is not None:
        config['path_root'] = path_root
    config_path.write_text(yaml.safe_dump(config))
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv('FOXES_TEST_DATA', str(tmp_path / 'external'))

    result = load_config(config_path)
    root = folder if path_root is None else folder.parent
    assert result['data']['aia_dir'] == str(root / 'images')
    assert result['data']['checkpoint_path'] == str(root / 'model.ckpt')
    assert result['data']['sxr_norm_path'] == absolute
    assert result['data']['aia_norm_path'] is None
    assert result['output_path'] == str(root / 'outputs/predictions.csv')
    assert result['plotting']['inference_patch_dir'] == str(tmp_path / 'external/maps')
    assert result['save_dir'] == str(Path.home() / 'foxes-downloads')
    assert result['weight_path'] == ''
    assert result['checkpoint']['monitor'] == 'val/total_loss'
    assert result['repo_id'] == config['repo_id']
    assert 'path_root' not in result


def test_download_and_evaluation_loaders_resolve_paths(tmp_path, monkeypatch):
    from download.hugging_face_data_download import load_config as load_download_config
    from forecasting.evaluation import load_evaluation_config

    config_path = tmp_path / 'workflow.yaml'
    config_path.write_text(yaml.safe_dump({
        'aia_dir': 'aia', 'model_predictions': {'main_model_csv': 'predictions.csv'},
    }))
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for loader in (load_download_config, load_evaluation_config):
        config = loader(config_path)
        assert config['aia_dir'] == str(tmp_path / 'aia')
        assert config['model_predictions']['main_model_csv'] == str(tmp_path / 'predictions.csv')


def test_training_inheritance_preserves_each_files_paths(tmp_path, monkeypatch):
    from training.train import load_training_config

    base_dir = tmp_path / 'base'
    override_dir = tmp_path / 'custom'
    base_dir.mkdir()
    override_dir.mkdir()
    (base_dir / 'train.yaml').write_text(yaml.safe_dump({
        'path_root': '..',
        'data': {'aia_dir': 'aia', 'sxr_dir': 'sxr', 'checkpoints_dir': 'weights'},
        'checkpoint': {'monitor': 'val/total_loss'},
    }))
    override = override_dir / 'train.yaml'
    override.write_text(yaml.safe_dump({
        'base_config': '../base/train.yaml',
        'path_root': 'run',
        'data': {'checkpoints_dir': 'weights'},
    }))
    monkeypatch.chdir(override_dir)

    config = load_training_config(override)
    assert config['data']['aia_dir'] == str(tmp_path / 'aia')
    assert config['data']['sxr_dir'] == str(tmp_path / 'sxr')
    assert config['data']['checkpoints_dir'] == str(override_dir / 'run/weights')
    assert config['checkpoint']['monitor'] == 'val/total_loss'


def test_bundled_configs_share_data_and_output_locations(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    inference = load_config(root / 'forecasting/inference_config.yaml')
    evaluation = load_config(root / 'forecasting/evaluation_config.yaml')
    download = load_config(root / 'download/hf_download_config.yaml')
    build = load_config(root / 'data/build_dataset_config.yaml')
    sdo = load_config(root / 'download/sdo_download_config.yaml')
    sxr = load_config(root / 'download/sxr_download_config.yaml')
    for filename in ('train_coupled_gaussian.yaml', 'train_deterministic.yaml'):
        train = load_config(root / 'training' / filename)
        assert train['data']['aia_dir'] == download['aia_dir'] == build['aia']['processed_dir']
        assert train['data']['sxr_dir'] == download['sxr_dir'] == build['output']['sxr_dir']
    assert sdo['download_dir'] == build['aia']['raw_dir']
    assert sxr['save_dir'] == build['sxr']['raw_dir']
    assert inference['data']['aia_dir'] == str(Path(download['aia_dir']) / 'test')
    assert inference['data']['sxr_dir'] == str(Path(download['sxr_dir']) / 'test')
    assert inference['data']['checkpoint_path'] == str(root / 'model_assets/UncertaintyModel.ckpt')
    assert evaluation['model_predictions']['main_model_csv'] == inference['output_path']


@pytest.mark.parametrize('contents, message', [
    ('[]', 'Config must be a mapping'),
    ('path_root: null', 'path_root must be'),
    ('data:\n  aia_dir: 123', 'data.aia_dir must be'),
])
def test_invalid_path_settings_report_the_setting(tmp_path, contents, message):
    config_path = tmp_path / 'bad.yaml'
    config_path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        load_config(config_path)

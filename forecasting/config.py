"""Load workflow YAML with paths independent of the working directory."""

import os
from pathlib import Path

import yaml


def _resolve_path(value, root, setting):
    if not isinstance(value, str):
        raise ValueError(f"{setting} must be a path string or null")
    if not value.strip():
        return value
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def load_config(config_path):
    """Resolve local file/directory settings relative to the YAML file.

    ``path_root`` optionally chooses another base, itself relative to the YAML
    file. Bundled configs set it to ``..`` to use the repository root. Absolute
    paths, ``~``, and environment variables are supported. ``base_config`` is
    left for the training loader to resolve relative to its declaring file.
    """
    config_path = Path(os.path.expandvars(os.fspath(config_path))).expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    root_value = config.get('path_root', '.')
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("path_root must be a nonempty path string")
    root = Path(_resolve_path(root_value, config_path.parent, 'path_root'))

    def resolve(value, key='', setting=''):
        if isinstance(value, dict):
            return {
                child_key: resolve(
                    child, child_key,
                    f'{setting}.{child_key}' if setting else child_key,
                )
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [resolve(child, key, f'{setting}[{i}]')
                    for i, child in enumerate(value)]
        if value is not None and (
            key.endswith(('_path', '_dir', '_file', '_csv'))
            or key in {'dirpath', 'resume_from_checkpoint'}
        ):
            return _resolve_path(value, root, setting)
        return value

    return resolve({key: value for key, value in config.items() if key != 'path_root'})

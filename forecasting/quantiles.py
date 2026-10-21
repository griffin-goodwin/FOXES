"""Names shared by quantile inference artifacts and their evaluator."""

import re


def quantile_label(level):
    """Round-trip the trained probability without merging nearby head labels."""
    return f'q{float(level)!r}'


def quantile_columns(level):
    raw = f'prediction_{quantile_label(level)}'
    return raw, f'{raw}_normalized'


def quantile_levels_from_columns(columns):
    """Read probabilities from raw CSV column names; reject incomplete schemas."""
    levels = []
    for column in columns:
        match = re.fullmatch(r'prediction_q(.+?)(?:_normalized)?', column)
        if match:
            try:
                level = float(match[1])
            except ValueError as exc:
                raise ValueError(f'Invalid quantile column: {column}') from exc
            if not 0 < level < 1 or column not in quantile_columns(level):
                raise ValueError(f'Invalid quantile column: {column}')
            levels.append(level)
    if not levels:
        return ()
    levels = tuple(sorted(set(levels)))
    required = {'prediction_normalized', 'groundtruth_normalized'}
    required.update(column for q in levels for column in quantile_columns(q))
    missing = required.difference(columns)
    if len(levels) != 5 or 0.5 not in levels or missing:
        raise ValueError('Incomplete quantile CSV: expected five probabilities '
                         f'including 0.5, with raw and normalized outputs; missing {sorted(missing)}')
    return levels


def quantile_interval_pairs(levels):
    """Pair lower/upper heads from the outside inward, preserving their identities."""
    return tuple(zip((q for q in levels if q < 0.5),
                     (q for q in reversed(levels) if q > 0.5)))

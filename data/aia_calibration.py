"""Local ITI AIA calibration tables, with a bundled snapshot for fresh installs."""

from importlib.resources import files
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
_CHANNELS = ["0094", "0131", "0171", "0193", "0211", "0304", "0335"]


def _read_table(source):
    """Read ITI's CSV format and reject unusable dates or correction factors."""
    table = pd.read_csv(source, parse_dates=["DATE"], index_col=0)
    factors = table[_CHANNELS].to_numpy(dtype=float)
    if (table.empty or not pd.api.types.is_integer_dtype(table.index.dtype)
            or not table.index.equals(pd.RangeIndex(len(table)))
            or not pd.api.types.is_datetime64_dtype(table["DATE"])
            or table["DATE"].isna().any()
            or not np.isfinite(factors).all() or (factors <= 0).any()):
        raise ValueError("Invalid ITI AIA calibration dates, row indices, or correction factors")
    return table


def get_auto_calibration_table():
    """Use a valid ITI cache, otherwise the packaged table, without downloading.

    Reading the existing cache preserves calibration for established installs.
    The fallback needs neither a network connection nor a writable home folder.
    ITI still selects the nearest dated row when applying the corrections.
    """
    cache = Path.home() / ".iti" / "sdo_autocal_table.csv"
    try:
        return _read_table(cache)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Cannot use ITI AIA calibration cache %s; using packaged table: %s", cache, exc)

    resource = files("data").joinpath("calibration").joinpath("sdo_autocal_table.csv")
    with resource.open("rb") as stream:
        return _read_table(stream)

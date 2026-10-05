"""Utilities for building an AIA-derived spatial SXR proxy.

The response-curve fit answers a narrow question: which nonnegative linear
combination of AIA temperature responses most closely follows the GOES/XRS-B
temperature response?  The empirical fit asks the analogous question on
paired, calibrated AIA DN/s images and observed GOES fluxes.

Neither fit is a DEM inversion.  Their channel weights are intended to build a
spatially normalized proxy, so an overall multiplicative calibration factor is
irrelevant to the eventual spatial prior.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.io import readsav
from scipy.optimize import linprog, nnls
from scipy.stats import pearsonr, spearmanr


AIA_WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)
CORONAL_WAVELENGTHS = (94, 131, 171, 193, 211, 335)

GOES_RESPONSE_RESOURCES = {
    # Matched to the bundled CHIANTI-v9 AIA response used below.
    "chianti_v9": {
        "url": (
            "https://sohoftp.nascom.nasa.gov/solarsoft/gen/idl/synoptic/goes/"
            "goes_chianti_resp_20200812.fits"
        ),
        "sha256": "59a94b5b2a625d02ff2b21958a21fb3452f4aef799d43031e3dd6b053c572e39",
    },
    # Useful for checking sensitivity to the current SolarSoft calibration.
    "latest": {
        "url": (
            "https://sohoftp.nascom.nasa.gov/solarsoft/gen/idl/synoptic/goes/"
            "goes_chianti_response_latest.fits"
        ),
        "sha256": "cb00c05850e3dc3bbd856eb07c1a372758d689d0845ee591d6e2531afeab0382",
    },
}
DEMREGPY_VERSION = "1.0.0"
DEMREGPY_PYPI_URL = f"https://pypi.org/pypi/demregpy/{DEMREGPY_VERSION}/json"
AIA_RESPONSE_MEMBER = "demregpy/data/aia_trespv9_en.dat"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None = None) -> Path:
    """Download atomically and optionally verify the file hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return destination
        destination.unlink()

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FOXES-calibration/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if expected_sha256 is not None and _sha256(temporary) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ensure_goes_response(
    cache_dir: str | Path, response_version: str = "chianti_v9"
) -> Path:
    """Return a verified local copy of the official SolarSoft GOES table."""
    override = os.environ.get("FOXES_GOES_RESPONSE_FILE")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FOXES_GOES_RESPONSE_FILE does not exist: {path}")
        return path
    if response_version not in GOES_RESPONSE_RESOURCES:
        raise ValueError(
            f"response_version must be one of {sorted(GOES_RESPONSE_RESOURCES)}"
        )
    resource = GOES_RESPONSE_RESOURCES[response_version]
    filename = Path(str(resource["url"])).name
    return _download(
        str(resource["url"]),
        Path(cache_dir) / filename,
        str(resource["sha256"]),
    )


def ensure_aia_response(cache_dir: str | Path) -> Path:
    """Extract demregpy's bundled CHIANTI-v9 AIA response without installing it."""
    override = os.environ.get("FOXES_AIA_RESPONSE_FILE")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FOXES_AIA_RESPONSE_FILE does not exist: {path}")
        return path

    cache_dir = Path(cache_dir)
    response_path = cache_dir / Path(AIA_RESPONSE_MEMBER).name
    if response_path.exists():
        return response_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DEMREGPY_PYPI_URL, headers={"User-Agent": "FOXES-calibration/1"}
    )
    with urllib.request.urlopen(request) as response:
        metadata = json.load(response)
    wheels = [
        item
        for item in metadata["urls"]
        if item["packagetype"] == "bdist_wheel"
        and item["filename"].endswith("py3-none-any.whl")
    ]
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one universal demregpy wheel, found {len(wheels)}")
    wheel_info = wheels[0]
    wheel_path = _download(
        wheel_info["url"],
        cache_dir / wheel_info["filename"],
        wheel_info["digests"]["sha256"],
    )
    with zipfile.ZipFile(wheel_path) as archive:
        payload = archive.read(AIA_RESPONSE_MEMBER)
    temporary = response_path.with_suffix(response_path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, response_path)
    return response_path


def load_aia_temperature_response(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load channel labels, log10(K), and response matrix from an IDL save file."""
    data = readsav(path, python_dict=True)
    channels = np.array(
        [int(value.decode("utf-8").removeprefix("A")) for value in data["channels"]]
    )
    log_temperature = np.asarray(data["logt"], dtype=np.float64)
    response = np.asarray(data["tr"], dtype=np.float64).T
    if response.shape != (log_temperature.size, channels.size):
        raise ValueError(
            f"Unexpected AIA response shape {response.shape}; expected "
            f"({log_temperature.size}, {channels.size})"
        )
    return channels, log_temperature, response


def load_goes_temperature_response(
    path: str | Path,
    satellite: int = 15,
    abundance: str = "coronal",
    a_primary: int = 1,
    b_primary: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load the XRS-B temperature response for one satellite/detector row.

    The returned curve is scaled to the flux produced by an emission measure of
    1e49 cm^-3, following the scaling used by sunkit-instruments.
    """
    abundance = abundance.lower()
    if abundance not in {"coronal", "photospheric"}:
        raise ValueError("abundance must be 'coronal' or 'photospheric'")
    table = fits.getdata(path, extension=1)
    names = set(table.names)
    if {"A_PRIM", "B_PRIM"}.issubset(names):
        mask = (
            (table["SAT"] == satellite)
            & (table["A_PRIM"] == a_primary)
            & (table["B_PRIM"] == b_primary)
        )
    elif "SECONDARY" in names:
        # The CHIANTI-v9 table has one primary row per satellite and additional
        # GOES-R secondary-detector rows marked with SECONDARY != 0.
        mask = (table["SAT"] == satellite) & (table["SECONDARY"] == 0)
    else:
        mask = table["SAT"] == satellite
    rows = table[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one GOES response row for satellite={satellite}, "
            f"detectors={a_primary}/{b_primary}; found {len(rows)}"
        )
    row = rows[0]
    flux_column = "FLONG_COR" if abundance == "coronal" else "FLONG_PHO"
    temperature_mk = np.asarray(row["TEMP_MK"], dtype=np.float64)
    log_temperature = np.log10(temperature_mk * 1e6)
    em_scale = 10.0 ** (49.0 - float(row["ALOG10EM"]))
    response = np.asarray(row[flux_column], dtype=np.float64) * em_scale
    metadata = {
        "satellite": int(row["SAT"]),
        "a_primary": int(row["A_PRIM"]) if "A_PRIM" in names else None,
        "b_primary": int(row["B_PRIM"]) if "B_PRIM" in names else None,
        "date": str(row["DATE"]),
        "version": str(row["VERSION"]),
        "method": str(row["METHOD"]),
        "abundance": abundance,
        "emission_measure_cm-3": 1e49,
    }
    return log_temperature, response, metadata


def fit_response_weights(
    aia_channels: Sequence[int],
    aia_log_temperature: np.ndarray,
    aia_response: np.ndarray,
    goes_log_temperature: np.ndarray,
    goes_response: np.ndarray,
    temperature_range: tuple[float, float] = (6.0, 8.0),
    goes_weight_power: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fit nonnegative channel weights to the GOES response curve.

    ``goes_weight_power=0`` weights all temperature samples equally.  The
    default softly emphasizes temperatures to which XRS-B is most sensitive
    without allowing the hottest endpoint to dominate the entire fit.
    """
    aia_channels = np.asarray(aia_channels, dtype=int)
    aia_log_temperature = np.asarray(aia_log_temperature, dtype=np.float64)
    aia_response = np.asarray(aia_response, dtype=np.float64)
    goes_log_temperature = np.asarray(goes_log_temperature, dtype=np.float64)
    goes_response = np.asarray(goes_response, dtype=np.float64)
    if aia_response.shape != (aia_log_temperature.size, aia_channels.size):
        raise ValueError("AIA response dimensions do not match its grids")
    low, high = temperature_range
    selected = (
        (goes_log_temperature >= low)
        & (goes_log_temperature <= high)
        & np.isfinite(goes_response)
        & (goes_response > 0)
    )
    logt = goes_log_temperature[selected]
    target = goes_response[selected]
    design = np.column_stack(
        [np.interp(logt, aia_log_temperature, aia_response[:, index])
         for index in range(aia_channels.size)]
    )
    column_scale = np.max(design, axis=0)
    if np.any(column_scale <= 0):
        raise ValueError("Every AIA response curve must be positive somewhere")
    target_scale = float(np.max(target))
    design_scaled = design / column_scale
    target_scaled = target / target_scale
    row_weight = np.power(target_scaled, goes_weight_power)
    coefficients, _ = nnls(
        design_scaled * row_weight[:, None],
        target_scaled * row_weight,
    )
    weights = coefficients * target_scale / column_scale
    approximation = design @ weights
    weight_series = pd.Series(weights, index=aia_channels, name="response_weight")
    curves = pd.DataFrame(
        {
            "log10_temperature_K": logt,
            "goes_xrsb": target,
            "aia_weighted_sum": approximation,
        }
    )
    return weight_series, curves


def _paired_names(aia_dir: Path, sxr_dir: Path) -> list[str]:
    sxr_names = {path.name for path in sxr_dir.glob("*.npy")}
    names = sorted(path.name for path in aia_dir.glob("*.npy") if path.name in sxr_names)
    if not names:
        raise FileNotFoundError(f"No paired .npy files in {aia_dir} and {sxr_dir}")
    return names


def load_integrated_sample(
    aia_root: str | Path,
    sxr_root: str | Path,
    split: str,
    sample_count: int,
    seed: int = 42,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Load a reproducible sample of calibrated images and integrate each channel."""
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be 'mean' or 'sum'")
    aia_dir = Path(aia_root) / split
    sxr_dir = Path(sxr_root) / split
    names = _paired_names(aia_dir, sxr_dir)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    if sample_count < len(names):
        indices = np.sort(rng.choice(len(names), size=sample_count, replace=False))
        names = [names[index] for index in indices]

    records: list[dict[str, object]] = []
    for name in names:
        image = np.asarray(np.load(aia_dir / name, allow_pickle=False), dtype=np.float64)
        if image.ndim != 3 or image.shape[0] != len(AIA_WAVELENGTHS):
            raise ValueError(f"Expected (7, H, W), found {image.shape} in {name}")
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError(f"No finite AIA pixels in {name}")
        if finite.min() >= -1.1 and finite.max() <= 1.1 and np.median(finite) < 0:
            raise ValueError(
                f"{aia_dir / name} looks like normalized [-1, 1] model input; "
                "use calibrated DN/s arrays instead"
            )
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0.0, None)
        sxr = float(np.asarray(np.load(sxr_dir / name, allow_pickle=False)).reshape(-1)[0])
        record: dict[str, object] = {"filename": name, "goes_xrsb_w_m2": sxr}
        for index, wavelength in enumerate(AIA_WAVELENGTHS):
            value = image[index].mean() if aggregation == "mean" else image[index].sum()
            record[f"aia_{wavelength}_{aggregation}_dn_s"] = float(value)
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    frame.attrs["aia_aggregation"] = aggregation
    return frame


def _feature_matrix(frame: pd.DataFrame, channels: Sequence[int]) -> np.ndarray:
    mean_columns = [f"aia_{channel}_mean_dn_s" for channel in channels]
    sum_columns = [f"aia_{channel}_sum_dn_s" for channel in channels]
    if all(column in frame for column in sum_columns):
        columns = sum_columns
    elif all(column in frame for column in mean_columns):
        columns = mean_columns
    else:
        raise KeyError(
            "Frame must contain either mean or sum AIA columns for every channel"
        )
    return frame[columns].to_numpy(dtype=np.float64)


def fit_integrated_weights(
    frame: pd.DataFrame,
    channels: Sequence[int] = AIA_WAVELENGTHS,
    loss: str = "l2",
) -> pd.Series:
    """Fit nonnegative AIA-to-GOES weights on spatially integrated images.

    ``loss='l2'`` is ordinary nonnegative least squares. ``loss='l1'`` is
    least absolute deviations, solved as a linear program. The L1 option means
    an L1 residual loss (MAE), not Lasso regularization on the coefficients.
    No intercept is fitted because a scalar intercept has no natural spatial
    allocation when these weights are later applied patch by patch.
    """
    channels = tuple(int(channel) for channel in channels)
    design = _feature_matrix(frame, channels)
    target = frame["goes_xrsb_w_m2"].to_numpy(dtype=np.float64)
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("Integrated AIA features and GOES targets must be finite")
    feature_scale = np.median(design, axis=0)
    feature_scale = np.where(feature_scale > 0, feature_scale, np.max(design, axis=0))
    if np.any(feature_scale <= 0):
        raise ValueError("Every fitted AIA channel must contain positive intensity")
    positive_target = target[target > 0]
    if positive_target.size == 0:
        raise ValueError("GOES targets must contain positive values")
    target_scale = float(np.median(positive_target))
    design_scaled = design / feature_scale
    target_scaled = target / target_scale

    if loss == "l2":
        coefficients, _ = nnls(design_scaled, target_scaled)
    elif loss == "l1":
        sample_count, feature_count = design_scaled.shape
        objective = np.concatenate(
            [np.zeros(feature_count), np.ones(sample_count)]
        )
        # |Xw-y| <= u becomes Xw-u <= y and -Xw-u <= -y.
        constraints = np.block(
            [
                [design_scaled, -np.eye(sample_count)],
                [-design_scaled, -np.eye(sample_count)],
            ]
        )
        bounds = np.concatenate([target_scaled, -target_scaled])
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=bounds,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"L1 regression failed: {result.message}")
        coefficients = result.x[:feature_count]
    else:
        raise ValueError("loss must be 'l2' or 'l1'")

    weights = coefficients * target_scale / feature_scale
    return pd.Series(weights, index=channels, name=f"integrated_{loss}")


def fit_empirical_weights(
    frame: pd.DataFrame,
    channels: Sequence[int] = CORONAL_WAVELENGTHS,
    relative_error: bool = False,
) -> pd.Series:
    """Fit nonnegative channel weights to paired full-disk AIA and GOES data."""
    channels = tuple(int(channel) for channel in channels)
    design = _feature_matrix(frame, channels)
    target = frame["goes_xrsb_w_m2"].to_numpy(dtype=np.float64)
    feature_scale = np.median(design, axis=0)
    feature_scale = np.where(feature_scale > 0, feature_scale, np.max(design, axis=0))
    if np.any(feature_scale <= 0):
        raise ValueError("Every fitted AIA channel must contain positive intensity")
    positive_target = target[target > 0]
    if positive_target.size == 0:
        raise ValueError("GOES targets must contain positive values")
    target_scale = float(np.median(positive_target))
    design_scaled = design / feature_scale
    target_scaled = target / target_scale
    if relative_error:
        floor = max(float(np.percentile(target_scaled[target_scaled > 0], 10)), 1e-6)
        row_weight = 1.0 / np.maximum(target_scaled, floor)
    else:
        row_weight = np.ones_like(target_scaled)
    coefficients, _ = nnls(
        design_scaled * row_weight[:, None],
        target_scaled * row_weight,
    )
    weights = coefficients * target_scale / feature_scale
    name = "empirical_relative" if relative_error else "empirical_linear"
    return pd.Series(weights, index=channels, name=name)


def proxy_values(frame: pd.DataFrame, weights: Mapping[int, float] | pd.Series) -> np.ndarray:
    channels = [int(channel) for channel in weights.keys()]
    coefficients = np.array([float(weights[channel]) for channel in channels])
    return _feature_matrix(frame, channels) @ coefficients


def fit_global_scale(
    frame: pd.DataFrame,
    weights: Mapping[int, float] | pd.Series,
    log_space: bool = True,
    flux_floor: float = 1e-10,
) -> float:
    """Fit one positive global scale without changing relative channel weights.

    A spatial prior is invariant to this scale. The default robust log-space
    fit therefore removes the arbitrary offset before comparing proxies over
    the many-decade GOES range. ``log_space=False`` gives an ordinary
    linear-flux least-squares scale when absolute-flux performance is the goal.
    """
    proxy = proxy_values(frame, weights)
    target = frame["goes_xrsb_w_m2"].to_numpy(dtype=np.float64)
    if not np.any(proxy > 0):
        return 0.0
    if log_space:
        valid = np.isfinite(proxy) & np.isfinite(target) & (proxy > 0) & (target > 0)
        if not np.any(valid):
            return 0.0
        log_ratio = np.log10(np.clip(target[valid], flux_floor, None)) - np.log10(
            np.clip(proxy[valid], flux_floor, None)
        )
        return float(np.power(10.0, np.median(log_ratio)))
    scale, _ = nnls(proxy[:, None], target)
    return float(scale[0])


def score_proxy(
    frame: pd.DataFrame,
    weights: Mapping[int, float] | pd.Series,
    global_scale: float = 1.0,
    flux_floor: float = 1e-10,
) -> dict[str, float]:
    target = frame["goes_xrsb_w_m2"].to_numpy(dtype=np.float64)
    prediction = proxy_values(frame, weights) * global_scale
    log_target = np.log10(np.clip(target, flux_floor, None))
    log_prediction = np.log10(np.clip(prediction, flux_floor, None))
    residual = log_prediction - log_target
    return {
        "mae_w_m2": float(np.mean(np.abs(prediction - target))),
        "rmse_w_m2": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "mae_dex": float(np.mean(np.abs(residual))),
        "rmse_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson_log": float(pearsonr(log_target, log_prediction).statistic),
        "spearman": float(spearmanr(target, prediction).statistic),
    }


def normalized_weights(weights: Mapping[int, float] | pd.Series) -> pd.Series:
    series = pd.Series(weights, dtype=np.float64)
    total = float(series.sum())
    return series / total if total > 0 else series


def patch_proxy_map(
    image: np.ndarray,
    weights: Mapping[int, float] | pd.Series,
    patch_size: int = 8,
) -> np.ndarray:
    """Return a sum-one patch map from a calibrated (7, H, W) AIA stack."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[0] != len(AIA_WAVELENGTHS):
        raise ValueError(f"Expected (7, H, W), found {image.shape}")
    _, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    proxy = np.zeros((height, width), dtype=np.float64)
    for channel, weight in weights.items():
        index = AIA_WAVELENGTHS.index(int(channel))
        proxy += float(weight) * np.clip(image[index], 0.0, None)
    patch_map = proxy.reshape(
        height // patch_size, patch_size, width // patch_size, patch_size
    ).sum(axis=(1, 3))
    total = float(patch_map.sum())
    return patch_map / total if total > 0 else patch_map

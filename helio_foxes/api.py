"""Fetch AIA, HARP, and GOES SXR observations and predict solar X-ray flux.

The public functions operate on channel-first arrays. Loading a checkpoint and
fetching data are separate so an application can reuse a model across requests.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import astropy.units as u
import drms
import pandas as pd
import numpy as np
from sunpy.coordinates import frames
from sunpy.net import attrs as a
from sunpy.net import jsoc as sunpy_jsoc

logger = logging.getLogger(__name__)


def _utc_datetime(value):
    """Interpret ISO strings and naive datetimes as UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime or ISO 8601 string")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ObservationMetadata:
    """Requested UTC time, channel order, and processed AIA crop geometry.

    ``timestamp`` is the requested time, not the actual exposure time of every
    channel. ``image_shape`` is the channel-first crop shape before model
    patching. If AIA fetching is disabled, it is the configured target shape.
    ``padding_factor`` describes the crop's padding beyond the solar radius.
    ``harp_bbox_expansion`` records the per-side HARP box expansion, including
    for HARP queries deferred until plotting.
    """

    timestamp: datetime | str
    image_shape: tuple[int, int, int]
    wavelengths: tuple[int, ...] = (94, 131, 171, 193, 211, 304, 335)
    padding_factor: float = 0.1
    harp_bbox_expansion: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "timestamp", _utc_datetime(self.timestamp))
        object.__setattr__(self, "image_shape", tuple(self.image_shape))
        object.__setattr__(self, "wavelengths", tuple(self.wavelengths))
        if len(self.image_shape) != 3 or any(
            not isinstance(n, (int, np.integer)) or n <= 0 for n in self.image_shape
        ):
            raise ValueError("image_shape must contain three positive integer dimensions")
        if len(self.wavelengths) != self.image_shape[0]:
            raise ValueError("wavelengths must match the image's channel count")
        if not np.isfinite(self.padding_factor) or self.padding_factor < 0:
            raise ValueError("padding_factor must be finite and nonnegative")
        if not np.isfinite(self.harp_bbox_expansion) or self.harp_bbox_expansion < 0:
            raise ValueError("harp_bbox_expansion must be finite and nonnegative")


@dataclass(frozen=True)
class Observation:
    """A processed observation, its metadata, and optional HARP/SXR data.

    ``data=None`` means AIA fetching was disabled.
    ``harp_regions=None`` means no HARP query was requested. An empty list
    means a query returned no usable regions, including query failures.
    ``sxr`` is the measured GOES result from :class:`SXRProcessor`, or
    ``None`` when disabled or unavailable.
    """

    metadata: ObservationMetadata
    data: np.ndarray | None
    harp_regions: list[dict] | None = None
    sxr: dict | None = None

    def __post_init__(self):
        if self.data is None:
            return
        object.__setattr__(self, "data", np.asarray(self.data))
        if self.data.shape != self.metadata.image_shape:
            raise ValueError("Observation data shape does not match its metadata")


class HARPProcessor:
    """Query HARP regions and project their boxes onto an AIA image crop.

    ``bbox_expansion`` adds this fraction of each projected box's width and
    height on each side, before clipping to the image. It defaults to zero.
    """

    _LIMB_BBOX_EDGE_SAMPLES = 25

    def __init__(self, *, email=None, resolution=512, padding_factor=0.1,
                 bbox_expansion=0.0):
        """Configure JSOC access and the AIA crop geometry."""
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if not np.isfinite(bbox_expansion) or bbox_expansion < 0:
            raise ValueError("bbox_expansion must be finite and nonnegative")
        self.email = email
        self.resolution = int(resolution)
        self.padding_factor = float(padding_factor)
        self.bbox_expansion = float(bbox_expansion)
        self.SHARP_SERIES = "hmi.sharp_720s"
        self.SEARCH_WINDOW = timedelta(minutes=15)
        self._QUERY_KEYS = "HARPNUM,NOAA_ARS,T_REC,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,RSUN_OBS,QUALITY"
        self._T_REC_FORMAT = "%Y.%m.%d_%H:%M:%S"

    def fetch_date(self, timestamp: datetime | str, jsoc_email: str | None = None) -> list[dict]:
        """Return raw JSOC rows from the nearest good-quality HARP sweep.

        ``timestamp`` accepts a datetime or ISO 8601 string, as in
        :meth:`AIAProcessor.fetch_date`. A timezone-free value is treated as
        UTC; an aware value is converted to UTC. For example,
        ``"2023-08-01T00:00:00Z"`` is accepted.

        Rows retain JSOC column names and values, including raw ``NOAA_ARS``.
        Visible regions also include ``top_pct``, ``left_pct``, ``width_pct``,
        and ``height_pct`` for the AIA crop. Query failures and absent usable
        regions return an empty list.
        """
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime or ISO 8601 string")
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        try:
            df = self._query_sharp(timestamp, jsoc_email or self.email)
            if df.empty:
                return []
            nearest = self._nearest_good_t_rec(df, timestamp)
            if nearest is None:
                return []
            sweep = df.loc[df["T_REC"] == nearest]
            good = sweep.loc[pd.to_numeric(sweep["QUALITY"], errors="coerce") == 0]
            regions = []
            for _, row in good.iterrows():
                try:
                    box = self._bbox_to_pixel_fractions(row)
                    if box is not None:
                        regions.append({**row.to_dict(), **box})
                except Exception as exc:  # One bad region need not hide the others.
                    logger.warning("HARP bbox transform failed for HARP %s: %s", row.get("HARPNUM"), exc)
            return regions
        except Exception as exc:  # HARP overlays are best-effort
            logger.warning("SHARP query failed: %s", exc)
            return []

    def _query_sharp(self, timestamp: datetime, jsoc_email: str) -> pd.DataFrame:
        """Query JSOC for HARP sweeps within ``SEARCH_WINDOW`` of ``timestamp``,
        via ``drms`` primarily, falling back to ``_query_sharp_via_sunpy`` if
        that raises (see its own docstring for what the fallback does and
        doesn't protect against). An empty-but-successful result is returned
        as-is, not treated as a failure -- that's real information (no HARP
        regions/no published data for this window), not something a second
        client would resolve differently.
        """
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        start = timestamp - self.SEARCH_WINDOW
        duration_s = int(2 * self.SEARCH_WINDOW.total_seconds())
        time_param = f"{start.strftime(self._T_REC_FORMAT)}_TAI/{duration_s}s"
        ds = f"{self.SHARP_SERIES}[][{time_param}]"

        logger.info("Querying JSOC via drms: %s", ds)
        try:  # pylint: disable=too-many-try-statements
            client = drms.Client(email=jsoc_email)
            return client.query(ds, key=self._QUERY_KEYS)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "drms query failed (%s); falling back to sunpy.net.jsoc", exc
            )
            return self._query_sharp_via_sunpy(timestamp, jsoc_email)

    def _query_sharp_via_sunpy(self, timestamp: datetime, jsoc_email: str) -> pd.DataFrame:
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        start = timestamp - self.SEARCH_WINDOW
        end = timestamp + self.SEARCH_WINDOW
        client = sunpy_jsoc.JSOCClient()
        response = client.search(
            a.Time(start.isoformat(), end.isoformat()),
            a.jsoc.Series(self.SHARP_SERIES),
            a.jsoc.Notify(jsoc_email),
        )
        if len(response) == 0:
            return pd.DataFrame()
        return response.show(*self._QUERY_KEYS.split(",")).to_pandas()

    def _parse_t_rec(self, t_rec: str) -> datetime:
        return datetime.strptime(t_rec.replace("_TAI", ""), self._T_REC_FORMAT)

    def _nearest_good_t_rec(self, df: pd.DataFrame, target: datetime) -> str | None:
        """Choose the closest sweep containing at least one QUALITY == 0 row."""
        if target.tzinfo is not None:
            target = target.astimezone(timezone.utc).replace(tzinfo=None)
        t_recs = sorted(
            df["T_REC"].unique(),
            key=lambda t: abs((self._parse_t_rec(str(t)) - target).total_seconds()),
        )
        for t_rec in t_recs:
            quality = df.loc[df["T_REC"] == t_rec, "QUALITY"]
            if (pd.to_numeric(quality, errors="coerce") == 0).any():
                return t_rec
        return None

    def _bbox_to_pixel_fractions(self, row: pd.Series) -> dict[str, float] | None:
        """Project a HARP heliographic box into display-space percentages.

        The HARP record's T_REC sets the observation time. Sample 25 points
        along each edge so curved projected edges near the limb are included.
        Invalid off-limb points are ignored; the result is clipped to the
        visible crop and is ``None`` when no area remains.
        """
        obstime = self._parse_t_rec(str(row["T_REC"]))
        rsun_obs = float(row["RSUN_OBS"])
        lon_min, lon_max = float(row["LON_MIN"]), float(row["LON_MAX"])
        lat_min, lat_max = float(row["LAT_MIN"]), float(row["LAT_MAX"])

        n = self._LIMB_BBOX_EDGE_SAMPLES
        lon_sweep = np.linspace(lon_min, lon_max, n)
        lat_sweep = np.linspace(lat_min, lat_max, n)
        lons = np.concatenate(
            [lon_sweep, lon_sweep, np.full(n, lon_min), np.full(n, lon_max)]
        )
        lats = np.concatenate(
            [np.full(n, lat_min), np.full(n, lat_max), lat_sweep, lat_sweep]
        )

        hg = frames.HeliographicStonyhurst(
            lons * u.deg, lats * u.deg, obstime=obstime
        )
        hpc = hg.transform_to(
            frames.Helioprojective(observer="earth", obstime=obstime)
        )
        tx_all = hpc.Tx.to_value(u.arcsec)
        ty_all = hpc.Ty.to_value(u.arcsec)

        valid = np.isfinite(tx_all) & np.isfinite(ty_all)
        if not valid.any():
            return None
        tx_vals, ty_vals = tx_all[valid], ty_all[valid]

        arcsec_per_pixel = (
            2 * (1 + self.padding_factor) * rsun_obs / self.resolution
        )
        center = self.resolution / 2
        px_left = center + min(tx_vals) / arcsec_per_pixel
        px_right = center + max(tx_vals) / arcsec_per_pixel
        # AIA array rows increase northward; display rows increase downward.
        disp_top = self.resolution - (center + max(ty_vals) / arcsec_per_pixel)
        disp_bottom = self.resolution - (center + min(ty_vals) / arcsec_per_pixel)

        expand_x = (px_right - px_left) * self.bbox_expansion
        expand_y = (disp_bottom - disp_top) * self.bbox_expansion
        px_left -= expand_x
        px_right += expand_x
        disp_top -= expand_y
        disp_bottom += expand_y

        px_left = max(0.0, min(px_left, self.resolution))
        px_right = max(0.0, min(px_right, self.resolution))
        disp_top = max(0.0, min(disp_top, self.resolution))
        disp_bottom = max(0.0, min(disp_bottom, self.resolution))
        if px_right <= px_left or disp_bottom <= disp_top:
            return None

        return {
            "top_pct": round(disp_top / self.resolution * 100, 2),
            "left_pct": round(px_left / self.resolution * 100, 2),
            "width_pct": round((px_right - px_left) / self.resolution * 100, 2),
            "height_pct": round((disp_bottom - disp_top) / self.resolution * 100, 2),
        }

class AIAProcessor:
    """Download AIA frames and assemble one stack for FOXES inference.

    The seven channels are ordered 94, 131, 171, 193, 211, 304, and 335 Å.
    Raw FITS downloads are kept only for the duration of :meth:`fetch_date`.
    ITI processing produces a channel-first ``float32`` array.

    Parameters
    ----------
    email : str or None, optional
        Registered JSOC email address used by the SDO downloader.
    resolution : int, default 512
        Height and width in pixels of each processed channel.
    tolerance : int, default 120
        Maximum time difference in seconds between the requested time and
        each selected AIA frame.
    n_workers : int, default 4
        Number of download threads for the seven wavelengths.
    aia_norm_path : str or pathlib.Path or None, optional
        AIA normalization artifact for the checkpoint. If supplied here,
        do not also supply it to :func:`load_model`.
    """

    WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)
    # AIAStackDataset uses ITI's NormalizeRadiusEditor default crop padding.
    PADDING_FACTOR = 0.1

    def __init__(self, *, email=None, resolution=512, tolerance=120,
                 n_workers=4, aia_norm_path=None):
        """Configure JSOC downloading and optional AIA normalization."""
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        from forecasting.dataset import AIANormTransform

        self.email = email
        self.resolution = int(resolution)
        self.tolerance = int(tolerance)
        self.n_workers = int(n_workers)
        self.transform = (
            AIANormTransform.from_file(aia_norm_path, self.WAVELENGTHS)
            if aia_norm_path is not None else None
        )

    def fetch_date(self, date):
        """Download and process all seven channels near a requested UTC time.

        Parameters
        ----------
        date : datetime.datetime or str
            Requested observation time. An ISO 8601 string with an offset or
            an aware datetime is converted to UTC. A value without timezone
            information is interpreted as UTC.

        Returns
        -------
        numpy.ndarray
            Finite ``float32`` AIA stack with shape
            ``(7, resolution, resolution)`` in :attr:`WAVELENGTHS` order.
            Optional AIA normalization is applied before returning.

        Raises
        ------
        TypeError
            If ``date`` is not a datetime or ISO 8601 string.
        FileNotFoundError
            If any wavelength has no downloaded frame in the search window.
        ValueError
            If ITI returns an unexpected shape or nonfinite values.

        Notes
        -----
        The raw FITS directory is deleted before the method returns, including
        when downloading or processing fails. The returned array is in memory.
        """
        from download.download_sdo import SDODownloader
        from data.convert_aia import AIAStackDataset

        date = _utc_datetime(date).replace(tzinfo=None)

        with TemporaryDirectory(prefix="foxes-aia-") as temporary_path:
            downloader = SDODownloader(
                base_path=temporary_path, email=self.email,
                wavelengths=[str(w) for w in self.WAVELENGTHS],
                n_workers=self.n_workers, tolerance=self.tolerance,
            )
            downloader.downloadDate(date)
            stamp = date.isoformat("T", timespec="seconds")
            paths = [
                Path(temporary_path) / str(w) / f"{stamp}.fits"
                for w in self.WAVELENGTHS
            ]
            missing = [w for w, path in zip(self.WAVELENGTHS, paths)
                       if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"AIA frames missing within {self.tolerance}s of {stamp} UTC "
                    f"for wavelengths {missing}"
                )
            dataset = AIAStackDataset(
                data=[[str(path)] for path in paths],
                wavelengths=self.WAVELENGTHS, resolution=self.resolution,
                ext=".fits", allow_errors=False,
            )
            stack = np.asarray(dataset[0], dtype=np.float32)
            expected = (len(self.WAVELENGTHS), self.resolution, self.resolution)
            if stack.shape != expected or not np.isfinite(stack).all():
                raise ValueError(f"Processed AIA stack must be finite with shape {expected}; got {stack.shape}")
            if self.transform is not None:
                import torch
                stack = self.transform(torch.from_numpy(stack)).numpy()
            return stack


class SXRProcessor:
    """Fetch GOES XRS and average satellites as in the training targets.

    ``tolerance`` is the maximum distance from the requested time, in seconds
    (default 120). ``satellite`` optionally restricts the GOES satellite
    number. Otherwise average all available satellites at the nearest usable
    sample timestamp, with equidistant timestamps favoring the earlier one.
    The arithmetic mean is taken in physical flux space, before any model
    log normalization, as in ``data.align_aia_sxr``.
    """

    def __init__(self, *, tolerance=120, satellite=None):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if satellite is not None and (
            isinstance(satellite, bool) or not isinstance(satellite, (int, np.integer))
            or satellite <= 0
        ):
            raise ValueError("satellite must be a positive GOES satellite number")
        self.tolerance = float(tolerance)
        self.satellite = int(satellite) if satellite is not None else None

    def fetch_date(self, date) -> dict | None:
        """Return measured XRS-A (0.5–4 Å) and XRS-B (1–8 Å) flux in W/m².

        Accepts the same datetime/ISO strings as :class:`AIAProcessor`.
        XRS-B must have a zero quality flag and finite positive flux, matching
        the training target's band selection. At the nearest valid timestamp,
        return the arithmetic mean ``xrsb_flux`` and the mean of available
        good-quality ``xrsa_flux`` values (or None if none are usable).
        ``timestamp`` is the actual UTC sample time; ``time_offset_seconds``
        is sample minus requested time. ``satellites`` lists the contributors
        and ``measurements`` retains their individual fluxes. Samples from
        different timestamps are never averaged together. Fluxes retain the
        science product's calibration; no interpolation or additional legacy
        scaling is applied. Missing data or query/processing failures
        return ``None``. Temporary downloads are removed before returning.
        """
        timestamp = _utc_datetime(date)
        try:
            from sunpy.net import Fido
            from sunpy.timeseries import TimeSeries

            window = timedelta(seconds=self.tolerance)
            query = [
                a.Time(timestamp - window, timestamp + window),
                a.Instrument("XRS"), a.Resolution("avg1m"),
            ]
            if self.satellite is not None:
                query.append(a.goes.SatelliteNumber(self.satellite))
            response = Fido.search(*query)
            if response.file_num == 0:
                return None
            candidates = []
            with TemporaryDirectory(prefix="foxes-sxr-") as directory:
                paths = Fido.fetch(response, path=str(Path(directory) / "{file}"), progress=False)
                if paths.errors:
                    logger.warning("Some GOES XRS downloads failed: %s", paths.errors)
                for path in paths:
                    try:
                        series = TimeSeries(path, source="XRS")
                        candidates.extend(self._samples(series, timestamp))
                    except ModuleNotFoundError:
                        raise  # Report environment problems with installation instructions below.
                    except Exception as exc:  # One corrupt file need not hide valid measurements.
                        logger.warning("GOES XRS processing failed for %s: %s", Path(path).name, exc)
            if not candidates:
                return None
            nearest = min(candidates, key=lambda row: (
                abs(row["time_offset_seconds"]), row["timestamp"],
            ))
            # A daily file can overlap another one; count each satellite once.
            contributors = {}
            for row in candidates:
                if row["timestamp"] == nearest["timestamp"]:
                    contributors.setdefault(row["satellite"], row)
            measurements = sorted(contributors.values(), key=lambda row: (
                -int(row["satellite"].split("-")[-1]),
            ))
            short_fluxes = [row["xrsa_flux"] for row in measurements if row["xrsa_flux"] is not None]
            return {
                "timestamp": nearest["timestamp"],
                "time_offset_seconds": nearest["time_offset_seconds"],
                "xrsb_flux": float(np.mean([row["xrsb_flux"] for row in measurements])),
                "xrsa_flux": float(np.mean(short_fluxes)) if short_fluxes else None,
                "satellites": [row["satellite"] for row in measurements],
                "measurements": measurements,
            }
        except ModuleNotFoundError as exc:
            if exc.name in {"h5netcdf", "h5py"}:
                import sys
                logger.warning(
                    "GOES XRS dependency %r is missing from %s. Install into this "
                    "environment with `%s -m pip install h5netcdf h5py`; in a notebook "
                    "run `%%pip install h5netcdf h5py` and restart the kernel.",
                    exc.name, sys.executable, sys.executable,
                )
            else:
                logger.warning("GOES XRS query failed: %s", exc)
            return None
        except Exception as exc:  # Supplemental observations are best-effort.
            logger.warning("GOES XRS query failed: %s", exc)
            return None

    def _samples(self, series, timestamp):
        """Keep valid samples in the window even for files spanning a day."""
        satellite = series.observatory
        if satellite is None:
            return []
        number = int(satellite.split("-")[-1])
        if self.satellite is not None and number != self.satellite:
            return []
        df = series.to_dataframe()
        times = pd.to_datetime(df.index, utc=True)
        offsets = np.asarray((times - pd.Timestamp(timestamp)).total_seconds())
        xrsa = np.asarray(series.quantity("xrsa").to_value(u.W / u.m**2), dtype=float)
        xrsb = np.asarray(series.quantity("xrsb").to_value(u.W / u.m**2), dtype=float)
        valid = (
            (np.abs(offsets) <= self.tolerance)
            & (pd.to_numeric(df["xrsb_quality"], errors="coerce").to_numpy() == 0)
            & np.isfinite(xrsb) & (xrsb > 0)
        )
        short_valid = (
            (pd.to_numeric(df["xrsa_quality"], errors="coerce").to_numpy() == 0)
            & np.isfinite(xrsa) & (xrsa > 0)
        )
        return [
            {
                "timestamp": times[index].to_pydatetime(),
                "satellite": f"GOES-{number}",
                "xrsa_flux": float(xrsa[index]) if short_valid[index] else None,
                "xrsb_flux": float(xrsb[index]),
                "time_offset_seconds": float(offsets[index]),
            }
            for index in np.flatnonzero(valid)
        ]


class ObservationProcessor:
    """Fetch AIA, HARP, and measured GOES SXR data for one UTC time.

    All sources are enabled by default. ``include_aia``, ``include_harps``, and
    ``include_sxr`` can be set here or overridden on :meth:`fetch_observation`. Disabled AIA
    fetching produces ``Observation.data=None``; disabled HARP fetching
    produces ``Observation.harp_regions=None``.

    ``email``, ``resolution``, ``tolerance``, ``n_workers``, and ``aia_norm_path``
    configure the component processors. AIA and HARP share the same crop
    geometry and email. HARP failures return an empty region list; AIA
    downloading or processing failures raise as in :class:`AIAProcessor`.
    ``harp_bbox_expansion`` adds that fraction on each side of projected HARP
    boxes, clipped to the image. Prediction sums use the expanded boxes.
    ``sxr_tolerance`` (seconds) and ``sxr_satellite`` configure the independent
    :class:`SXRProcessor`. Disabled or unavailable SXR produces ``sxr=None``.
    """

    def __init__(self, *, email=None, resolution=512, tolerance=120,
                 n_workers=4, aia_norm_path=None, include_aia=True,
                 include_harps=True, harp_bbox_expansion=0.0,
                 include_sxr=True, sxr_tolerance=120, sxr_satellite=None):
        self.include_aia = include_aia
        self.include_harps = include_harps
        self.include_sxr = include_sxr
        self.aia_processor = AIAProcessor(
            email=email, resolution=resolution, tolerance=tolerance,
            n_workers=n_workers, aia_norm_path=aia_norm_path,
        )
        self.harp_processor = HARPProcessor(
            email=email, resolution=self.aia_processor.resolution,
            padding_factor=self.aia_processor.PADDING_FACTOR,
            bbox_expansion=harp_bbox_expansion,
        )
        self.sxr_processor = SXRProcessor(tolerance=sxr_tolerance, satellite=sxr_satellite)

    def fetch_observation(self, date, *, include_aia=None, include_harps=None,
                          include_sxr=None) -> Observation:
        """Fetch enabled sources for a datetime or ISO 8601 UTC timestamp.

        Per-call toggles override the processor's defaults for this call only.
        Metadata and any fetched HARP/SXR data travel with the returned
        observation through prediction and plotting. AIA data is required
        for prediction. Raw downloads are removed before returning.
        """
        timestamp = _utc_datetime(date)
        if include_aia is None:
            include_aia = self.include_aia
        if include_harps is None:
            include_harps = self.include_harps
        if include_sxr is None:
            include_sxr = self.include_sxr
        stack = self.aia_processor.fetch_date(timestamp) if include_aia else None
        metadata = ObservationMetadata(
            timestamp=timestamp,
            image_shape=stack.shape if stack is not None else (
                len(self.aia_processor.WAVELENGTHS),
                self.aia_processor.resolution, self.aia_processor.resolution,
            ),
            wavelengths=self.aia_processor.WAVELENGTHS,
            padding_factor=self.aia_processor.PADDING_FACTOR,
            harp_bbox_expansion=self.harp_processor.bbox_expansion,
        )
        regions = self.harp_processor.fetch_date(timestamp) if include_harps else None
        sxr = self.sxr_processor.fetch_date(timestamp) if include_sxr else None
        return Observation(
            metadata=metadata, data=stack, harp_regions=regions, sxr=sxr,
        )


@dataclass(frozen=True)
class Prediction:
    """FOXES output for one AIA stack.

    Attributes
    ----------
    metadata : ObservationMetadata or None
        Input observation metadata, preserved for automatic HARP overlays.
        The original input array is not retained.
    flux : float
        Predicted full-disk soft X-ray flux in W/m².
    flux_map : numpy.ndarray
        Two-dimensional patch contributions in W/m². Its dimensions are the
        checkpoint's patch grid, which is smaller than the input image.
    intervals : dict or None
        For Gaussian checkpoints, ``"68"`` and ``"95"`` map to lower and
        upper flux bounds in W/m². For pinball checkpoints, keys give the
        nominal percentage between paired quantile heads (e.g. ``"90"`` for
        q0.05/q0.95); endpoints are not sorted and may cross.
        ``None`` for deterministic checkpoints.
    quantiles : dict or None
        Trained probability to full-disk prediction in W/m² for pinball models.
        ``flux`` and ``flux_map`` use the median head.
    quantile_flux_maps : dict or None
        Trained probability to its original spatial contribution map in W/m².
        These are allocations of global quantiles, not calibrated patch quantiles.
    patch_uncertainty_map : numpy.ndarray or None
        For Gaussian checkpoints, the square root of each patch's contribution
        to global flux variance, in W/m². These are variance attributions,
        not calibrated patch-level error bars.
    harp_regions : list[dict] or None
        Per-HARP prediction results. Each dict contains ``harpnum``,
        ``flux_sum`` in W/m², and ``top_pct``, ``left_pct``, ``width_pct``,
        ``height_pct`` locations. Original JSOC fields, including heliographic
        longitude/latitude bounds, are preserved. Flux sums include positive
        patch contributions whose centers fall inside each box. Overlapping
        boxes can count the same patch. ``None`` means HARPs were not fetched;
        an empty list means no usable regions were returned.
        ``flux_sum_uncertainty`` is sqrt(sum(patch_uncertainty_map**2))
        inside the same box, in W/m², or None for deterministic predictions.
        This follows the model's additive variance attribution; regional
        uncertainty calibration is not established.
    sxr : dict or None
        Measured GOES fluxes and sample provenance, preserved from Observation.
        Compare ``sxr["xrsb_flux"]`` with the predicted ``flux``. This measured
        value is not an input to the model. None when disabled or unavailable.
    """

    metadata: ObservationMetadata | None = field(default=None, kw_only=True)
    flux: float
    flux_map: np.ndarray
    intervals: dict[str, tuple[float, float]] | None = None
    patch_uncertainty_map: np.ndarray | None = None
    harp_regions: list[dict] | None = None
    sxr: dict | None = None
    quantiles: dict[float, float] | None = None
    quantile_flux_maps: dict[float, np.ndarray] | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "harp_regions", _harp_region_fluxes(
                self.flux_map, self.harp_regions, self.patch_uncertainty_map,
            ),
        )


class Predictor:
    """Keep a FOXES checkpoint loaded for repeated predictions.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Path to a compatible Lightning ``.ckpt`` or serialized model file.
    device : {"auto", "cpu", "cuda", "mps"}, default "auto"
        Inference device. ``"auto"`` selects an available accelerator.
    aia_norm_path : str or pathlib.Path or None, optional
        Matching AIA normalization artifact, if the checkpoint was trained
        with one. Do not apply that artifact in :class:`AIAProcessor` too.

    Notes
    -----
    Create the predictor once when the UI starts and reuse it for requests.
    """

    def __init__(self, checkpoint_path, *, device="auto", aia_norm_path=None):
        """Load the checkpoint and its optional input normalization."""
        from forecasting.inference import _unwrap_model, load_model_from_config
        from forecasting.dataset import AIANormTransform

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = load_model_from_config({
            "data": {"checkpoint_path": str(path)}, "device": device,
        })
        self.base_model = _unwrap_model(self.model)
        self.device = next(self.model.parameters()).device
        self.transform = (
            AIANormTransform.from_file(aia_norm_path)
            if aia_norm_path is not None else None
        )
        self.model_kwargs = dict(self.base_model.hparams["model_kwargs"])

    def predict(self, image, *, timestamp=None):
        """Predict global SXR flux and patch contributions from one AIA stack.

        Parameters
        ----------
        image : Observation or array-like
            Finite channel-first processed AIA stack in 94, 131, 171, 193,
            211, 304, 335 Å order. Shape is ``(7, H, W)`` with ``H`` and
            ``W`` determined by the checkpoint's patch size and grid.
            Observation metadata is copied into the returned prediction.
        timestamp : datetime or str or None, optional
            Requested UTC time for a plain array. Its crop is assumed to use
            the standard AIA channel order and 0.1 padding. For other crops,
            supply an Observation with explicit ObservationMetadata.

        Returns
        -------
        Prediction
            Full-disk flux, patch flux map, and optional Gaussian or quantile outputs.

        Raises
        ------
        ValueError
            If the image shape is incompatible or values are nonfinite.
        """
        import torch
        from forecasting.inference import uncertainty_from_prediction

        if isinstance(image, Observation) and image.data is None:
            raise ValueError("Observation has no AIA data; fetch with include_aia=True before prediction")
        metadata = image.metadata if isinstance(image, Observation) else None
        array = np.asarray(image.data if isinstance(image, Observation) else image)
        patch_size = int(self.model_kwargs["patch_size"])
        patch_count = int(self.model_kwargs["num_patches"])
        grid = int(np.sqrt(patch_count))
        channels = int(self.model_kwargs["num_channels"])
        expected = (channels, grid * patch_size, grid * patch_size)
        if array.shape != expected:
            raise ValueError(f"Expected processed AIA shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("AIA image contains non-finite values")
        if metadata is not None:
            if metadata.wavelengths != AIAProcessor.WAVELENGTHS:
                raise ValueError("Observation wavelengths must be in FOXES AIA channel order")
            if timestamp is not None and _utc_datetime(timestamp) != metadata.timestamp:
                raise ValueError("timestamp conflicts with the observation metadata")
        elif timestamp is not None:
            metadata = ObservationMetadata(timestamp=timestamp, image_shape=array.shape)
        tensor = torch.as_tensor(np.array(array, dtype=np.float32, copy=True))
        if self.transform is not None:
            tensor = self.transform(tensor)
        tensor = tensor.permute(1, 2, 0).unsqueeze(0).to(self.device)
        quantiles = quantile_flux_maps = None
        with torch.inference_mode():
            outputs = self.model(tensor, return_attention=False)
            if getattr(self.base_model, 'uncertainty_kind', None) == 'quantile':
                from forecasting.quantiles import quantile_interval_pairs
                raw, _, all_patches = outputs
                levels = self.base_model.quantiles
                median_index = levels.index(0.5)
                flux = raw[:, median_index]
                patches = all_patches[:, :, median_index]
                raw_cpu = raw[0].detach().cpu().numpy()
                maps_cpu = all_patches[0].reshape(grid, grid, len(levels)).detach().cpu().numpy()
                quantiles = dict(zip(levels, map(float, raw_cpu)))
                quantile_flux_maps = {q: maps_cpu[:, :, i] for i, q in enumerate(levels)}
                intervals = {
                    f'{100 * (upper - lower):g}': (quantiles[lower], quantiles[upper])
                    for lower, upper in quantile_interval_pairs(levels)
                }
                patch_uncertainty = None
            elif getattr(self.base_model, "predicts_uncertainty", False):
                flux, variance, patches, patch_variance = outputs
                intervals_raw = uncertainty_from_prediction(
                    flux, variance, self.base_model.sxr_norm,
                )
                intervals = {
                    level: (
                        float(intervals_raw[f"lower_{level}"].item()),
                        float(intervals_raw[f"upper_{level}"].item()),
                    ) for level in ("68", "95")
                }
                patch_uncertainty = torch.sqrt(
                    patch_variance.clamp_min(0)
                ).reshape(grid, grid).detach().cpu().numpy()
            else:
                flux, patches = outputs
                intervals = None
                patch_uncertainty = None
        flux_map = patches.reshape(grid, grid).detach().cpu().numpy()
        return Prediction(
            metadata=metadata,
            flux=float(flux.item()), flux_map=flux_map,
            intervals=intervals, patch_uncertainty_map=patch_uncertainty,
            quantiles=quantiles, quantile_flux_maps=quantile_flux_maps,
            harp_regions=image.harp_regions if isinstance(image, Observation) else None,
            sxr=image.sxr if isinstance(image, Observation) else None,
        )


def load_model(checkpoint_path, *, device="auto", aia_norm_path=None):
    """Load a checkpoint for repeated calls to :func:`predict`.

    Parameters are the same as for :class:`Predictor`.

    Returns
    -------
    Predictor
        Loaded model and optional AIA transform.
    """
    return Predictor(checkpoint_path, device=device, aia_norm_path=aia_norm_path)


def predict(image, *, model, timestamp=None):
    """Run one processed AIA stack through a loaded FOXES model.

    Parameters
    ----------
    image : Observation or array-like
        Channel-first AIA stack; see :meth:`Predictor.predict` for its shape
        and channel order.
    model : Predictor
        Predictor returned by :func:`load_model`.
    timestamp : datetime or str or None, optional
        Requested time for an array input. Observation inputs carry their own
        timestamp and crop metadata. See :meth:`Predictor.predict`.

    Returns
    -------
    Prediction
        Full-disk flux, flux map, and optional uncertainty in W/m².
    """
    return model.predict(image, timestamp=timestamp)


def _log_limits(positive, vmin, vmax):
    """Resolve positive color limits, expanding a constant automatic range."""
    low = float(positive.min()) if vmin is None else float(vmin)
    high = float(positive.max()) if vmax is None else float(vmax)
    if vmin is None and vmax is None and low == high:
        low, high = low / 10, high * 10
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or low >= high:
        raise ValueError("vmin and vmax must be finite, positive, and vmin < vmax")
    return low, high


def _harp_box_patch_sum(flux_map, *, top, left, width, height):
    """Sum positive patch flux where patch centers lie inside a HARP box."""
    rows, cols = flux_map.shape
    x_centers = (np.arange(cols) + 0.5) / cols * 100
    y_centers = (rows - np.arange(rows) - 0.5) / rows * 100
    inside_x = (x_centers >= left) & (x_centers <= left + width)
    inside_y = (y_centers >= top) & (y_centers <= top + height)
    return float(np.maximum(flux_map[np.ix_(inside_y, inside_x)], 0).sum())


def _harp_region_fluxes(flux_map, regions, patch_uncertainty_map=None):
    """Copy HARP boxes and sum their flux and additive variance contributions."""
    flux_map = np.asarray(flux_map)
    patch_variance = None
    if patch_uncertainty_map is not None:
        uncertainty = np.asarray(patch_uncertainty_map, dtype=float)
        if uncertainty.ndim != 2 or uncertainty.shape != flux_map.shape:
            raise ValueError("patch_uncertainty_map must be 2D and match flux_map shape")
        if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
            raise ValueError("patch_uncertainty_map must be finite and nonnegative")
        patch_variance = np.square(uncertainty)
    if regions is None:
        return None
    results = []
    for region in regions:
        top, left, width, height = (
            float(region[key]) for key in
            ("top_pct", "left_pct", "width_pct", "height_pct")
        )
        if not np.isfinite([top, left, width, height]).all() or width <= 0 or height <= 0:
            raise ValueError("HARP boxes need finite positive width and height percentages")
        harpnum = region.get("harpnum", region.get("HARPNUM"))
        if harpnum is None:
            raise ValueError("HARP regions need a harpnum or HARPNUM identifier")
        results.append({
            **region,
            "harpnum": int(harpnum),
            "top_pct": top, "left_pct": left,
            "width_pct": width, "height_pct": height,
            "flux_sum": _harp_box_patch_sum(
                flux_map, top=top, left=left, width=width, height=height,
            ),
            "flux_sum_uncertainty": float(np.sqrt(_harp_box_patch_sum(
                patch_variance, top=top, left=left, width=width, height=height,
            ))) if patch_variance is not None else None,
        })
    return results


def _harp_rank_color(rank):
    """Return a fixed color for a zero-based flux rank, independent of count."""
    from matplotlib.colors import hsv_to_rgb, to_rgb

    palette = (
        "#56B4E9", "#E69F00", "#009E73", "#CC79A7",
        "#0072B2", "#D55E00", "#F0E442", "#999999",
    )
    if rank < len(palette):
        return to_rgb(palette[rank])
    return tuple(hsv_to_rgb([(rank * 0.61803398875) % 1, 0.6, 0.85]))


def plot_flux_map(prediction, *, ax=None, cmap="inferno", vmin=None, vmax=None,
                  show_harp_boxes=False, harp_regions=None, timestamp=None,
                  jsoc_email=None, min_harp_flux=1e-7, harp_label_fontsize=6,
                  show_harp_uncertainty=True, harp_label_offsets=None,
                  harp_label_max_shift=80):
    """Plot patch flux contributions with a logarithmic color scale.

    For a Prediction, the title shows its predicted FOXES flux with the
    model's full-disk uncertainty when available, and the measured SXR-B
    satellite mean when available, all in W/m². The uncertainty is the
    square root of summed patch variances, not a calibrated error bar.
    Missing values are omitted. Plain map arrays do not add a flux title.

    Parameters
    ----------
    prediction : Prediction or array-like
        A FOXES result or a two-dimensional patch flux map in W/m².
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created if omitted.
    cmap : str, default "inferno"
        Matplotlib colormap name.
    vmin, vmax : float or None, optional
        Positive limits for the logarithmic color scale in W/m². By default,
        use the smallest and largest positive values in the map.
    show_harp_boxes : bool, default False
        Draw HARP outlines labeled with their numbers and flux sums, with
        fixed labels anchored to box corners. Box and label colors are fixed
        by descending flux rank, independent of the number of HARPs. Sums include
        positive patch contributions whose centers fall inside each box.
    harp_regions : list[dict] or None, optional
        Pre-fetched regions from :meth:`HARPProcessor.fetch_date`.
        Defaults to the prediction's regions. If neither has been fetched
        and ``show_harp_boxes`` is true, regions are queried from JSOC. An
        empty list suppresses the query, including one carried by a prediction.
    timestamp : datetime or str or None, optional
        AIA observation time, in the same format accepted by
        :meth:`AIAProcessor.fetch_date`. Defaults to the prediction's metadata.
        Required for an automatic HARP query when metadata is absent.
    jsoc_email : str or None, optional
        JSOC email for the automatic HARP query.
    min_harp_flux : float, default 1e-7
        Minimum HARP flux sum in W/m² to show in the map. Regions
        below this value are hidden; prediction results remain complete.
    harp_label_fontsize : float, default 6
        Fixed HARP label size in points, independent of box size.
    show_harp_uncertainty : bool, default True
        Add a third label line, ``± uncertainty``, when patch uncertainties
        are available. This is sqrt(sum(sigma_patch**2)) within the HARP box,
        in W/m², using the model's additive variance attribution. Disabling
        this affects labels only; prediction results retain uncertainties.
    harp_label_offsets : dict[int, tuple[float, float]] or None, optional
        Manual (dx, dy) label offsets in points, keyed by HARP number. Positive
        dx moves right; positive dy moves up, relative to the chosen box corner.
        Custom offsets add a matching connector. Defaults to a three-point
        diagonal gap outside a box corner, preferring the side toward the image center.
        The overlap pass tries all four corners before small four-point nudges.
        Offsets are starting positions; shifted labels get connectors.
        Placement is calculated once; replot after substantial size changes.
    harp_label_max_shift : float, default 80
        Maximum automatic nudge from a corner, in points. Set to zero to fix
        labels at their preferred corner. Automatic positions stay outside
        their HARP box and inside the map, unless a manual offset was supplied.
        If no clear position is found within this limit, retain the starting position; crowded maps
        may still need manual offsets or a higher min_harp_flux.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the map. Its figure also has a flux colorbar.

    Raises
    ------
    ValueError
        If the map is not finite and two-dimensional, has no positive values,
        or the color limits are invalid. Nonpositive cells are masked.
    """
    from matplotlib import pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    flux_map = np.asarray(
        prediction.flux_map if isinstance(prediction, Prediction) else prediction
    )
    if flux_map.ndim != 2 or not np.isfinite(flux_map).all():
        raise ValueError("flux_map must be a finite 2D array")
    positive = flux_map[flux_map > 0]
    if positive.size == 0:
        raise ValueError("flux_map must contain positive contributions")
    low, high = _log_limits(positive, vmin, vmax)
    if not np.isfinite(min_harp_flux) or min_harp_flux < 0:
        raise ValueError("min_harp_flux must be finite and nonnegative (W/m²)")
    if not np.isfinite(harp_label_fontsize) or harp_label_fontsize <= 0:
        raise ValueError("harp_label_fontsize must be finite and positive")
    if not np.isfinite(harp_label_max_shift) or harp_label_max_shift < 0:
        raise ValueError("harp_label_max_shift must be finite and nonnegative")
    if show_harp_boxes and harp_regions is None and isinstance(prediction, Prediction):
        harp_regions = prediction.harp_regions
    if show_harp_boxes and harp_regions is None:
        metadata = prediction.metadata if isinstance(prediction, Prediction) else None
        if metadata is not None:
            if timestamp is not None and _utc_datetime(timestamp) != metadata.timestamp:
                raise ValueError("timestamp conflicts with the prediction metadata")
            timestamp = metadata.timestamp
        if timestamp is None:
            raise ValueError("timestamp is required to query HARP regions")
        crop = {} if metadata is None else {
            "resolution": metadata.image_shape[-1],
            "padding_factor": metadata.padding_factor,
            "bbox_expansion": metadata.harp_bbox_expansion,
        }
        harp_regions = HARPProcessor(email=jsoc_email, **crop).fetch_date(timestamp)
    regions = sorted(
        (region for region in _harp_region_fluxes(
            flux_map, harp_regions,
            prediction.patch_uncertainty_map if isinstance(prediction, Prediction) else None,
        )
         if region["flux_sum"] >= min_harp_flux),
        key=lambda region: (-region["flux_sum"], region["harpnum"]),
    ) if show_harp_boxes else []
    if ax is None:
        _, ax = plt.subplots()
    artist = ax.imshow(
        np.ma.masked_less_equal(flux_map, 0),
        origin="lower", cmap=cmap,
        norm=LogNorm(vmin=low, vmax=high)
    )
    ax.set(xlabel="X (Patches)", ylabel="Y (Patches)")
    if isinstance(prediction, Prediction):
        title = []
        sxr_b = prediction.sxr.get("xrsb_flux") if prediction.sxr is not None else None
        if sxr_b is not None and np.isfinite(sxr_b):
            title.append(f"Observed SXR-B: {sxr_b:.2e} W/m²")
        if prediction.flux is not None and np.isfinite(prediction.flux):
            predicted = f"Predicted FOXES: {prediction.flux:.2e}"
            if prediction.patch_uncertainty_map is not None:
                uncertainty = np.sqrt(np.sum(np.square(prediction.patch_uncertainty_map)))
                if np.isfinite(uncertainty):
                    predicted += f" ± {uncertainty:.2e}"
            title.append(f"{predicted} W/m²")
        ax.set_title("\n".join(title))
    ax.figure.colorbar(artist, ax=ax, label="Flux contribution (W m$^{-2}$)")
    if regions:
        rows, cols = flux_map.shape
        colors = [_harp_rank_color(rank) for rank in range(len(regions))]
        offsets = harp_label_offsets or {}
        # Render once so label sizes can be measured in screen pixels.
        ax.figure.canvas.draw()
        renderer = ax.figure._get_renderer()
        map_bounds = ax.get_window_extent(renderer)
        occupied = []
        occupied_regions = []
        for rank, (region, color) in enumerate(zip(regions, colors)):
            top, left, width, height = (
                float(region[key]) for key in
                ("top_pct", "left_pct", "width_pct", "height_pct")
            )
            x = -0.5 + left / 100 * cols
            y = -0.5 + (100 - top - height) / 100 * rows
            box_width = width / 100 * cols
            box_height = height / 100 * rows
            outline = Rectangle(
                (x, y), box_width, box_height,
                fill=False, edgecolor=color, linewidth=1.8,
            )
            outline.set_path_effects([
                patheffects.Stroke(linewidth=3.5, foreground="black"),
                patheffects.Normal(),
            ])
            ax.add_patch(outline)
            region_bounds = outline.get_window_extent(renderer)
            occupied_regions.append(region_bounds)
            text = f'#{region["harpnum"]}\n{region["flux_sum"]:.2e}'
            if show_harp_uncertainty and region["flux_sum_uncertainty"] is not None:
                text += f'\n± {region["flux_sum_uncertainty"]:.2e}'
            # Text extends diagonally outward from each corner.
            horizontal = [(x, "right", -3), (x + box_width, "left", 3)]
            vertical = [(y, "top", -3), (y + box_height, "bottom", 3)]
            if left + width / 2 < 50:
                horizontal.reverse()
            if top + height / 2 > 50:
                vertical.reverse()
            placements = [((cx, cy), (dx, dy), ha, va)
                          for cy, va, dy in vertical for cx, ha, dx in horizontal]
            harpnum = region["harpnum"]
            corner, inset, ha, va = placements[0]
            if harpnum in offsets:
                placements = [(corner, offsets[harpnum], ha, va)]
            elif harp_label_max_shift == 0:
                placements = placements[:1]
            label = ax.annotate(
                text, xy=corner, xytext=offsets.get(harpnum, inset), textcoords="offset points",
                ha=ha, va=va,
                color=color, fontsize=harp_label_fontsize, fontweight="bold",
                zorder=3 + len(regions) - rank,  # Stronger labels draw above later connectors.
                bbox={"facecolor": "black", "alpha": .5, "edgecolor": "none", "pad": 1},
                arrowprops={"arrowstyle": "-", "color": color},
            )
            # Try the four corners first, then small nudges from the preferred corner.
            dx, dy = placements[0][1]
            for step in range(4, int(harp_label_max_shift) + 1, 4):
                placements.extend((corner, position, ha, va) for position in
                                  [(dx, dy + step), (dx, dy - step),
                                   (dx + step, dy), (dx - step, dy)])
            chosen = placements[0]  # Retain the start if no nearby space is free.
            for anchor, position, horizontal_align, vertical_align in placements:
                label.xy = anchor
                label.set(position=position, ha=horizontal_align, va=vertical_align)
                label.update_positions(renderer)
                label.update_bbox_position_size(renderer)
                bounds = label.get_bbox_patch().get_window_extent(renderer).padded(2)
                within_map = (map_bounds.contains(*bounds.p0) and map_bounds.contains(*bounds.p1))
                if not within_map and harpnum not in offsets:
                    continue
                outside_x = bounds.x1 < region_bounds.x0 or bounds.x0 > region_bounds.x1
                outside_y = bounds.y1 < region_bounds.y0 or bounds.y0 > region_bounds.y1
                if not (outside_x and outside_y) and harpnum not in offsets:
                    continue
                if harpnum not in offsets and any(bounds.overlaps(box) for box in occupied_regions):
                    continue
                if not any(bounds.overlaps(other) for other in occupied):
                    chosen = (anchor, position, horizontal_align, vertical_align)
                    break
            label.xy = chosen[0]
            label.set(position=chosen[1], ha=chosen[2], va=chosen[3])
            label.update_positions(renderer)
            label.update_bbox_position_size(renderer)
            bounds = label.get_bbox_patch().get_window_extent(renderer).padded(2)
            occupied.append(bounds)
            shifted = chosen != placements[0]
            label.arrow_patch.set_visible(shifted or harpnum in offsets)
    return ax

def plot_uncertainty_map(prediction, *, ax=None, cmap="magma", vmin=None, vmax=None):
    """Plot patch uncertainty contributions with a logarithmic color scale.

    Parameters
    ----------
    prediction : Prediction or array-like
        A FOXES result or a two-dimensional patch uncertainty map in W/m².
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created if omitted.
    cmap : str, default "magma"
        Matplotlib colormap name.
    vmin, vmax : float or None, optional
        Positive limits for the logarithmic color scale in W/m². By default,
        use the smallest and largest positive values in the map.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the map. Its figure also has an uncertainty colorbar.

    Raises
    ------
    ValueError
        If the map is not finite and two-dimensional, has no positive values,
        or the color limits are invalid. Nonpositive cells are masked.
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    uncertainty_map = np.asarray(
        prediction.patch_uncertainty_map if isinstance(prediction, Prediction) else prediction
    )
    if uncertainty_map.ndim != 2 or not np.isfinite(uncertainty_map).all():
        raise ValueError("uncertainty_map must be a finite 2D array")
    positive = uncertainty_map[uncertainty_map > 0]
    if positive.size == 0:
        raise ValueError("uncertainty_map must contain positive contributions")
    low, high = _log_limits(positive, vmin, vmax)
    if ax is None:
        _, ax = plt.subplots()
    artist = ax.imshow(
        np.ma.masked_less_equal(uncertainty_map, 0),
        origin="lower", cmap=cmap,
        norm=LogNorm(vmin=low, vmax=high)
    )
    ax.set(xlabel="X (Patches)", ylabel="Y (Patches)")
    ax.figure.colorbar(artist, ax=ax, label="Patch uncertainty contribution (W m$^{-2}$)")
    return ax

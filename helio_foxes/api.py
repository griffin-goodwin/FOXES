"""Fetch AIA, HARP, and GOES SXR observations and predict solar X-ray flux.

The public functions operate on channel-first arrays. Loading a checkpoint and
fetching data are separate so an application can reuse a model across requests.
"""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
import logging
import json
import os
import re
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import astropy.units as u
import drms
import pandas as pd
import numpy as np
from sunpy.coordinates import frames
from sunpy.net import attrs as a
from sunpy.net import jsoc as sunpy_jsoc
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def parse_timestamp(value):
    """Parse a complete date/time and return an aware UTC datetime.

    Accept datetime/date, numpy.datetime64, ISO 8601, year-first slash/dot
    dates (including ``2023.08.01_12:00:00_UTC``), and named-month dates.
    Date-only values mean midnight; naive values mean UTC. Numeric dates
    such as ``01/08/2023`` are rejected rather than guessing day/month order.
    Unix numbers, partial dates, relative dates, and timezone abbreviations
    other than UTC/GMT are not supported. Use an explicit offset for zones.

    Parameters
    ----------
    value : datetime.datetime, datetime.date, numpy.datetime64, or str
        A complete date, optionally with a time. Examples: "2023-08-01",
        "2023-08-01T12:00:00Z", "2023/08/01 12:00",
        "2023.08.01_12:00:00_UTC", "August 1, 2023 12:00 UTC", or
        "2023-08-01T08:00:00-04:00". Date-only inputs mean midnight; naive
        inputs mean UTC. Numeric Unix timestamps, NaT, incomplete dates such
        as "August 2023", ambiguous "01/08/2023", and non-UTC/GMT timezone
        abbreviations such as EST are rejected.

    Returns
    -------
    datetime.datetime
        Timezone-aware UTC time.
    """
    original = value
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("timestamp cannot be NaT")
        value = np.datetime_as_string(value, unit="us")
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        from dateutil.parser import parse, UnknownTimezoneWarning

        value = value.strip()
        try:
            if re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}", value):
                raise ValueError("ambiguous numeric date; use year-month-day")
            if value.endswith("_UTC"):
                value = value[:-4] + "+00:00"
            value = re.sub(r"^(\d{4})[/.](\d{2})[/.](\d{2})", r"\1-\2-\3", value)
            value = value.replace("_", "T")
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                def resolve_zone(name, offset):
                    if name not in (None, "UTC", "GMT"):
                        raise ValueError(f"timezone abbreviation {name!r} is ambiguous; use a numeric offset")
                    return timezone(timedelta(seconds=offset)) if offset is not None else None

                with warnings.catch_warnings():
                    warnings.simplefilter("error", UnknownTimezoneWarning)
                    first = parse(value, default=datetime(2000, 1, 1), tzinfos=resolve_zone)
                    second = parse(value, default=datetime(2004, 2, 2), tzinfos=resolve_zone)
                if first != second:
                    raise ValueError("a complete year, month, and day are required")
                value = first
        except (ValueError, OverflowError, UnknownTimezoneWarning) as exc:
            raise ValueError(
                f"Invalid timestamp {original!r}: {exc}. Use a complete date such as "
                "'2023-08-01 12:00:00', '2023/08/01', or 'August 1, 2023'; "
                "use Z or a numeric UTC offset for timezone information."
            ) from exc
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime, date, numpy.datetime64, or date string")
    if pd.isna(value):
        raise ValueError("timestamp cannot be NaT")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


_utc_datetime = parse_timestamp


def _encode_observation_metadata(value):
    """Keep metadata JSON-safe without pickling arrays or datetime objects."""
    if isinstance(value, np.generic):
        return _encode_observation_metadata(value.item())
    if isinstance(value, datetime):
        return {"__foxes_datetime__": _utc_datetime(value).isoformat()}
    if isinstance(value, float) and not np.isfinite(value):
        return {"__foxes_float__": str(value)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Observation metadata dictionary keys must be strings")
        return {key: _encode_observation_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_encode_observation_metadata(item) for item in value]
    if value is pd.NA:
        return None
    return value


def _decode_observation_metadata(value):
    if set(value) == {"__foxes_datetime__"}:
        return _utc_datetime(value["__foxes_datetime__"])
    if set(value) == {"__foxes_float__"}:
        return float(value["__foxes_float__"])
    return value


@dataclass(frozen=True)
class ObservationMetadata:
    """Requested UTC time, channel order, and processed AIA crop geometry.

    ``timestamp`` is the requested time, not the actual exposure time of every
    channel. ``image_shape`` is the channel-first crop shape before model
    patching. If AIA fetching is disabled, it is the configured target shape.
    ``padding_factor`` describes the crop's padding beyond the solar radius.
    ``harp_bbox_expansion`` records the per-side HARP box expansion, including
    for HARP queries deferred until plotting.

    Parameters
    ----------
    timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str
        Complete observation date/time accepted by parse_timestamp, e.g.
        "2023/08/01 12:00" or "2023-08-01T12:00:00Z". Naive inputs mean UTC.
    image_shape : sequence of three positive int
        Channel-first (C, H, W), e.g. (7, 512, 512); never (H, W, C).
    wavelengths : sequence of int, optional
        Channel wavelengths in Angstroms, one per channel, in array order.
        FOXES prediction requires (94, 131, 171, 193, 211, 304, 335).
    padding_factor : float, default 0.1
        Nonnegative crop padding beyond the solar radius; not a pixel count.
    harp_bbox_expansion : float, default 0.0
        Nonnegative fraction of box width/height added on each side; 0.1 means
        10 percent per side, not 0.1 pixels.
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

    Parameters
    ----------
    metadata : ObservationMetadata
        Timestamp, channel order, and geometry describing this exact array.
    data : array-like or None
        Channel-first (C, H, W) data matching metadata.image_shape, or None
        when AIA is disabled. Use from_array for finite float32 validation.
    harp_regions : list of dict or None, optional
        Each region needs HARPNUM or harpnum (integer-like) and numeric
        top_pct, left_pct, width_pct, height_pct, in percent of the exact crop.
        Origin is the top-left display corner; dimensions must be positive.
        For example: {"HARPNUM": 1234, "top_pct": 10, "left_pct": 20,
        "width_pct": 15, "height_pct": 12}. None means unqueried; [] means
        explicitly no usable regions. Prediction does not query HARPs itself.
    sxr : dict or None, optional
        Measured flux/provenance from SXRProcessor.fetch_date; xrsb_flux and
        xrsa_flux are in W/m². This is comparison data, not a model input.
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

    @classmethod
    def from_array(cls, data, *, timestamp, wavelengths=(94, 131, 171, 193, 211, 304, 335),
                   padding_factor=0.1, harp_regions=None, sxr=None):
        """Wrap an already preprocessed channel-first AIA stack, without downloading.

        This does not calibrate, crop, resize, or normalize the input. Match
        training preprocessing and apply the checkpoint's normalization once.
        Optional HARP boxes use percentages of this exact crop, measured from
        its top-left display corner. ``None`` leaves HARPs unqueried; ``[]``
        explicitly supplies no regions. The predictor validates model shape
        and channel order. The returned observation owns a float32 copy.

        Parameters
        ----------
        data : array-like
            Finite real numeric (C, H, W) stack, usually (7, 512, 512), in channel
            order. Nested numeric lists or a NumPy array work. Channels-last arrays
            are not transposed automatically. The returned array is a float32 copy.
        timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str
            Required keyword. Formats accepted by parse_timestamp, e.g.
            "2023/08/01 12:00" or "2023-08-01T12:00:00Z". Naive values mean UTC.
        wavelengths : sequence of int, optional
            One Angstrom wavelength per channel. Prediction requires the default
            (94, 131, 171, 193, 211, 304, 335); one-channel snapshots cannot be predicted.
        padding_factor : float, default 0.1
            Nonnegative padding fraction matching the preprocessing crop.
        harp_regions : list of dict or None, optional
            HARP identifiers and top/left/width/height percentages; see Observation.
        sxr : dict or None, optional
            Optional measured SXR flux/provenance; see Observation.

        Returns
        -------
        Observation
            Validated input container, ready for a shape-compatible predictor.
        """
        array = np.asarray(data)
        if array.ndim != 3 or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError("data must be a finite real numeric channel-first (C, H, W) array")
        with np.errstate(over="ignore"):
            array = array.astype(np.float32, copy=True)
        if not np.isfinite(array).all():
            raise ValueError("data exceeds the float32 range")
        return cls(
            metadata=ObservationMetadata(timestamp, array.shape, wavelengths, padding_factor),
            data=array, harp_regions=harp_regions, sxr=sxr,
        )

    @classmethod
    def from_npy(cls, path, *, timestamp, **kwargs):
        """Load one processed ``.npy`` stack; see :meth:`from_array`.

        Parameters
        ----------
        path : str or pathlib.Path
            Local .npy file containing exactly one numeric (C, H, W) stack, e.g.
            "processed_aia.npy". Pickled/object arrays are disabled. Complete
            .npz observation archives must instead use Observation.load.
        timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str
            Required keyword; e.g. "2023-08-01 12:00". Naive values mean UTC.
        **kwargs
            wavelengths, padding_factor, harp_regions, and sxr, with the same
            types and meanings as from_array. Timestamp is not read from filename.

        Returns
        -------
        Observation
            Processed array plus the explicitly supplied metadata.
        """
        return cls.from_array(np.load(path, allow_pickle=False), timestamp=timestamp, **kwargs)

    def save(self, path, *, overwrite=False):
        """Save a complete observation as a compressed NumPy ``.npz`` archive.

        Stores ``data`` (if present) and scalar JSON ``metadata``, including
        the UTC time, crop geometry, channel order, HARP rows, and SXR data.
        No pickle is used. Arrays retain their dtype and values. A temporary
        file in the destination directory is published only after a complete
        write. Existing files raise FileExistsError unless overwrite=True.
        Returns the saved Path; load with :meth:`load`, without preprocessing
        or normalizing again. ``from_npy`` remains for plain array-only files.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination ending in .npz, e.g. "saved/observation.npz". Parent
            directories are created and a leading ~ is expanded. This is a complete
            archive, unlike a plain .npy array.
        overwrite : bool, default False
            True atomically replaces the file; False raises FileExistsError if it
            exists. Range iterators separately reuse existing saved observations.

        Returns
        -------
        pathlib.Path
            Destination of the completed archive.
        """
        path = Path(path).expanduser()
        if path.suffix != ".npz":
            raise ValueError("Observation.save requires a .npz path")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        if self.data is not None and (
            self.data.dtype.kind not in "iuf" or not np.isfinite(self.data).all()
        ):
            raise ValueError("Observation data must be finite real numeric values to save")
        payload = {
            "schema_version": 1, "has_data": self.data is not None,
            "metadata": asdict(self.metadata),
            "harp_regions": self.harp_regions, "sxr": self.sxr,
        }
        encoded = json.dumps(_encode_observation_metadata(payload), allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=".foxes-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                arrays = {"metadata": np.asarray(encoded)}
                if self.data is not None:
                    arrays["data"] = self.data
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path):
        """Reload a complete archive made by :meth:`save`, with pickle disabled.

        Parameters
        ----------
        path : str or pathlib.Path
            Existing complete .npz archive produced by Observation.save, e.g.
            "saved/observation.npz". Not a raw FITS file or array-only .npy file.

        Returns
        -------
        Observation
            Restored numeric array, datetime metadata, and optional HARP/SXR data.
            No preprocessing, normalization, network access, or pickle execution.
        """
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(archive["metadata"].item(), object_hook=_decode_observation_metadata)
            if payload.get("schema_version") != 1:
                raise ValueError("Unsupported observation archive schema_version")
            has_data = payload["has_data"]
            if not isinstance(has_data, bool) or has_data != ("data" in archive):
                raise ValueError("Observation archive data does not match its metadata")
            data = archive["data"] if has_data else None
        metadata = ObservationMetadata(**payload["metadata"])
        if data is not None and (data.dtype.kind not in "iuf" or not np.isfinite(data).all()):
            raise ValueError("Observation archive data must be finite real numeric values")
        return cls(metadata=metadata, data=data, harp_regions=payload["harp_regions"], sxr=payload["sxr"])


def load_observations(directory, *, progress=False):
    """Yield saved ``.npz`` observations in filename order, without network access.

    Iterator-generated filenames sort by requested UTC time. Reads one file
    at a time so it can be passed directly to ``model.predict_series``.
    A missing directory or one with no .npz files raises FileNotFoundError.
    Invalid archives raise instead of silently dropping observations.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory of complete observation .npz archives, e.g. "saved_observations".
        A leading ~ is expanded. Files are sorted lexicographically; the range
        iterator's timestamp filenames therefore yield chronological order.
        Do not mix normalization artifacts or other .npz files in this directory.
    progress : bool, default False
        Enable a tqdm loading bar in a terminal or notebook.

    Yields
    ------
    Observation
        One restored observation at a time; accepts direct use as input to
        model.predict_series(load_observations("saved_observations")).
    """
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Observation directory does not exist: {directory}")
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No observation .npz archives found in {directory}")
    for path in tqdm(paths, desc="Loading observations", unit="observation", disable=not progress):
        yield Observation.load(path)


class HARPProcessor:
    """Query HARP regions and project their boxes onto an AIA image crop.

    ``bbox_expansion`` adds this fraction of each projected box's width and
    height on each side, before clipping to the image. It defaults to zero.
    """

    _LIMB_BBOX_EDGE_SAMPLES = 25

    def __init__(self, *, email=None, resolution=512, padding_factor=0.1,
                 bbox_expansion=0.0):
        """Configure JSOC access and the AIA crop geometry.

        Parameters
        ----------
        email : str or None, optional
            Registered JSOC email; e.g. "you@example.com".
        resolution : int, default 512
            Positive image side length in pixels, matching the processed AIA crop.
        padding_factor : float, default 0.1
            Crop padding fraction beyond the solar radius, not pixels.
        bbox_expansion : float, default 0.0
            Finite nonnegative fractional expansion per box side; 0.1 means 10%.
        """
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

        ``timestamp`` accepts all parse_timestamp formats, as in
        :meth:`AIAProcessor.fetch_date`. A timezone-free value is treated as
        UTC; an aware value is converted to UTC. For example,
        ``"2023-08-01T00:00:00Z"`` is accepted.

        Rows retain JSOC column names and values, including raw ``NOAA_ARS``.
        Visible regions also include ``top_pct``, ``left_pct``, ``width_pct``,
        and ``height_pct`` for the AIA crop. Query failures and absent usable
        regions return an empty list.

        Parameters
        ----------
        timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str
            Complete time accepted by parse_timestamp, e.g. "2023/08/01 12:00",
            "August 1, 2023 12:00 UTC", or "2023-08-01T08:00:00-04:00".
            Date-only values mean midnight; naive values mean UTC. JSOC's internal
            TAI record syntax is not a supported public timestamp format.
        jsoc_email : str or None, optional
            Per-call registered JSOC email; None uses the processor's email.

        Returns
        -------
        list of dict
            Good-quality HARP rows and projected crop percentages; [] on absent
            usable regions or query failure. Invalid input timestamps still raise.
        """
        timestamp = _utc_datetime(timestamp).replace(tzinfo=None)
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
        #Building the full box by concatenating the four edges in order: bottom, top, left, right.
        lons = np.concatenate(
            [lon_sweep, lon_sweep, np.full(n, lon_min), np.full(n, lon_max)]
        )
        lats = np.concatenate(
            [np.full(n, lat_min), np.full(n, lat_max), lat_sweep, lat_sweep]
        )
        #Convert To Proper Frame
        hg = frames.HeliographicStonyhurst(
            lons * u.deg, lats * u.deg, obstime=obstime
        )
        #Transform
        hpc = hg.transform_to(
            frames.Helioprojective(observer="earth", obstime=obstime)
        )
        #Convert To Arcseconds
        tx_all = hpc.Tx.to_value(u.arcsec)
        ty_all = hpc.Ty.to_value(u.arcsec)

        valid = np.isfinite(tx_all) & np.isfinite(ty_all)
        if not valid.any():
            return None
        tx_vals, ty_vals = tx_all[valid], ty_all[valid]
        #Calculate Arcseconds Per Pixel And Crop Edges In Pixels (1.1 Solar Radii)
        arcsec_per_pixel = (
            2 * (1 + self.padding_factor) * rsun_obs / self.resolution
        )
        #Get Pixel Coordinates Of The Box Edges In The AIA Crop
        center = self.resolution / 2
        px_left = center + min(tx_vals) / arcsec_per_pixel
        px_right = center + max(tx_vals) / arcsec_per_pixel
        # AIA array rows increase northward; display rows increase downward.
        disp_top = self.resolution - (center + max(ty_vals) / arcsec_per_pixel)
        disp_bottom = self.resolution - (center + min(ty_vals) / arcsec_per_pixel)

        #Add and expansion to the harps in X / Y
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
        #Return BBox Coordinates As Percentages Of The AIA Crop
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
        """Configure JSOC downloading and optional AIA normalization.

        Parameters
        ----------
        email : str or None, optional
            Registered JSOC email, e.g. "you@example.com".
        resolution : int, default 512
            Positive processed height and width in pixels.
        tolerance : int, default 120
            Search half-width in seconds around the requested timestamp.
        n_workers : int, default 4
            Number of concurrent download threads.
        aia_norm_path : str or pathlib.Path or None, optional
            Checkpoint-matched .npz or .csv AIA normalization artifact. Apply it
            here or in Predictor, never both; None leaves this transform disabled.
        """
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
        date : datetime.datetime, datetime.date, numpy.datetime64, or str
            Complete date/time accepted by parse_timestamp, e.g.
            "2023-08-01T12:00:00Z", "2023/08/01 12:00",
            "2023.08.01_12:00:00_UTC", or "August 1, 2023".
            Date-only values mean midnight; timezone-free values mean UTC.
            Numeric Unix timestamps and ambiguous dates are not accepted.

        Returns
        -------
        numpy.ndarray
            Finite ``float32`` AIA stack with shape
            ``(7, resolution, resolution)`` in :attr:`WAVELENGTHS` order.
            Optional AIA normalization is applied before returning.

        Raises
        ------
        TypeError
            If ``date`` is not a supported datetime/date/string value.
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

        date = _utc_datetime(date).replace(tzinfo=None)

        with TemporaryDirectory(prefix="foxes-aia-") as temporary_path:
            #Download the seven AIA FITS files in parallel into a temporary directory
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
            return self.process_files(paths)

    def process_files(self, paths):
        """Preprocess seven local FITS files in :attr:`WAVELENGTHS` order.

        Uses the same ITI calibration/crop/resize pipeline as downloads.
        Caller-owned files are never removed. Returns a processed array;
        wrap it with ``Observation.from_array(stack, timestamp=...)``.

        Parameters
        ----------
        paths : iterable of str or pathlib.Path
            Exactly seven existing local FITS file paths in wavelength order
            (94, 131, 171, 193, 211, 304, 335). Supply a list/tuple of paths, not a
            directory name or a single string. Example:
            [Path("fits") / f"{w}.fits" for w in AIAProcessor.WAVELENGTHS].

        Returns
        -------
        numpy.ndarray
            Finite float32 array shaped (7, resolution, resolution). Optional
            processor normalization is applied once; local FITS files are preserved.
        """
        from data.convert_aia import AIAStackDataset

        paths = [Path(path) for path in paths]
        if len(paths) != len(self.WAVELENGTHS):
            raise ValueError(f"Provide seven FITS files in wavelength order {self.WAVELENGTHS}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        dataset = AIAStackDataset(
            data=[[str(path)] for path in paths], wavelengths=self.WAVELENGTHS,
            resolution=self.resolution, ext=".fits", allow_errors=False,
        )
        stack = np.asarray(dataset[0], dtype=np.float32)
        expected = (len(self.WAVELENGTHS), self.resolution, self.resolution)
        if stack.shape != expected or not np.isfinite(stack).all():
            raise ValueError(f"Processed AIA stack must be finite with shape {expected}; got {stack.shape}")
        if self.transform is not None:
            import torch
            stack = self.transform(torch.from_numpy(stack)).numpy()
        return stack

    def _iter_range(self, timestamps, *, on_error, progress=False):
        """Preprocess selected frames from one bulk JSOC metadata query."""
        from download.download_sdo import SDODownloader

        with TemporaryDirectory(prefix="foxes-aia-range-") as directory:
            #Download bulk AIA FITS files in parallel into a temporary directory, then process them.
            downloader = SDODownloader(
                base_path=directory, email=self.email,
                wavelengths=[str(w) for w in self.WAVELENGTHS],
                n_workers=self.n_workers, tolerance=self.tolerance,
            )
            for timestamp, paths in downloader.iter_download_range(
                [stamp.replace(tzinfo=None) for stamp in timestamps], on_error=on_error, progress=progress,
            ):
                try:
                    stack = self.process_files(paths)
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    logger.warning("Skipping AIA preprocessing at %s: %s", timestamp, exc)
                else:
                    yield timestamp.replace(tzinfo=timezone.utc), stack
                finally:
                    for path in paths:
                        Path(path).unlink(missing_ok=True)


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
        """Configure the GOES XRS lookup.

        Parameters
        ----------
        tolerance : float, default 120
            Finite positive search half-width in seconds.
        satellite : positive int or None, optional
            GOES satellite number, e.g. 18 (not "GOES-18"). None combines valid
            satellites at the nearest usable timestamp as described by the class.
        """
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

        Accepts all date/time formats described by :func:`parse_timestamp`.
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

        Parameters
        ----------
        date : datetime.datetime, datetime.date, numpy.datetime64, or str
            Complete date/time accepted by parse_timestamp. For example,
            "2023-08-01T12:00:00Z", "2023/08/01 12:00", or "August 1, 2023".
            Date-only values mean midnight; naive inputs mean UTC.

        Returns
        -------
        dict or None
            Physical xrsa_flux/xrsb_flux in W/m², an actual UTC sample timestamp,
            offset in seconds, satellites, and individual measurements; None on
            unavailable data/query failure. Invalid input timestamps still raise.
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
        """Configure observation sources and shared crop geometry.

        Parameters
        ----------
        email : str or None, optional
            Registered JSOC email shared by AIA and HARP queries.
        resolution : int, default 512
            Positive processed AIA height/width in pixels; match the checkpoint.
        tolerance : int, default 120
            AIA search half-width in seconds; independent of sxr_tolerance.
        n_workers : int, default 4
            AIA download thread count.
        aia_norm_path : str or pathlib.Path or None, optional
            Matching .npz/.csv AIA normalization artifact. Do not also normalize
            in the predictor when this is supplied.
        include_aia, include_harps, include_sxr : bool, default True
            Default source toggles, overridable per fetch/range call.
        harp_bbox_expansion : float, default 0.0
            Nonnegative box expansion fraction per side; 0.1 means 10% per side.
        sxr_tolerance : float, default 120
            Finite positive GOES lookup half-width in seconds.
        sxr_satellite : positive int or None, optional
            Restrict GOES to a number such as 18, or use all valid satellites.
        """
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
        """Fetch enabled sources for any supported complete date/time format.

        Per-call toggles override the processor's defaults for this call only.
        Metadata and any fetched HARP/SXR data travel with the returned
        observation through prediction and plotting. AIA data is required
        for prediction. Raw downloads are removed before returning.

        Parameters
        ----------
        date : datetime.datetime, datetime.date, numpy.datetime64, or str
            Complete time accepted by parse_timestamp, e.g. "2023/08/01 12:00",
            "2023-08-01T12:00:00Z", or "August 1, 2023". Date-only values mean
            midnight and naive values mean UTC; numeric epoch times are not accepted.
        include_aia, include_harps, include_sxr : bool or None, optional
            True enables the source, False disables it, and None uses the constructor
            default. The constructor settings are not modified.

        Returns
        -------
        Observation
            Requested UTC time and processed array, plus enabled HARP/SXR data.
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

    def iter_observations(self, start, end, *, cadence="1h", on_error="raise",
                          save_dir=None, overwrite=False, progress=False, **kwargs):
        """Download/preprocess a UTC range one observation at a time.

        Includes start and any cadence-aligned time <= end. Cadence accepts
        a pandas duration string (``'12min'``), timedelta, or seconds as a
        number. Source toggles are passed to :meth:`fetch_observation`.
        AIA uses one bulk JSOC record/segment query across the padded range,
        followed by downloads of the selected frames. HARP and SXR queries
        currently run separately for each timestamp. Large spans can have
        large metadata responses; split long jobs into smaller ranges.
        Defaults to raising AIA errors. ``on_error='skip'`` logs failures
        with their requested timestamps and continues. HARP/SXR retain
        their single-observation best-effort behavior. No interpolation is
        performed; nearby requests can select the same exposure. Raw files
        are temporary. Use this iterator to avoid retaining all AIA arrays.
        ``save_dir`` writes each successful observation as a complete .npz
        archive before yielding it; see ``Observation.save`` and
        ``load_observations``. Saving is opt-in and never changes the data.
        Existing archives are loaded and yielded without downloading,
        preprocessing, or HARP/SXR queries. Only missing times enter the
        bulk AIA request. ``overwrite=True`` explicitly refreshes all times.
        Cache load/save errors raise even with ``on_error='skip'``. Cached
        metadata and enabled-source contents are reused as saved; use
        overwrite=True after changing preprocessing or query settings.
        ``progress=True`` shows tqdm during the bulk query and subsequent
        AIA downloading/preprocessing (including skipped attempts). Without
        AIA it tracks fetching the supplemental observations instead.

        Parameters
        ----------
        start, end : datetime.datetime, datetime.date, numpy.datetime64, or str
            Inclusive range bounds accepted by parse_timestamp, e.g.
            start="2023/08/01 00:00", end="2023-08-01T06:00:00Z". Both use UTC
            when timezone-free. End must be at or after start. End is only yielded
            when it lands on the cadence grid anchored at start.
        cadence : str, datetime.timedelta, pandas.Timedelta, int, or float, default "1h"
            Positive duration in whole seconds. Examples: "12min", "1h",
            timedelta(minutes=12), or 720 (numeric values mean seconds, not minutes).
        on_error : {"raise", "skip"}, default "raise"
            Raise individual download/preprocessing failures or log and skip them.
            Initial bulk-query errors and archive load/save failures always raise.
        save_dir : str or pathlib.Path or None, optional
            Directory for complete timestamped .npz observations, e.g. "saved_aia".
            Existing files are reused without data queries or preprocessing unless
            overwrite=True. None disables persistent saving/reuse.
        overwrite : bool, default False
            True downloads/processes again and atomically replaces saved observations.
        progress : bool, default False
            Show tqdm for remaining work or for loading a completely saved range.
        **kwargs
            include_aia, include_harps, include_sxr: bool or None, each with the
            same source-toggle rules as fetch_observation.

        Yields
        ------
        Observation
            Successful saved/new observations in requested timestamp order. Merely
            creating this generator does no work: consume it with a for loop, list,
            or model.predict_series to start downloading/saving.

        Examples
        --------
        >>> observations = processor.iter_observations(
        ...     "2023/08/01", "2023-08-01 01:00", cadence="12min",
        ...     save_dir="saved_aia", progress=True,
        ... )
        >>> # Consume observations with model.predict_series(observations).
        """
        start, end = _utc_datetime(start), _utc_datetime(end)
        if end < start:
            raise ValueError("end must be at or after start")
        if on_error not in {"raise", "skip"}:
            raise ValueError("on_error must be 'raise' or 'skip'")
        try:
            if isinstance(cadence, bool):
                raise ValueError("boolean cadence")
            step = pd.to_timedelta(cadence, unit="s") if isinstance(cadence, (int, float)) else pd.Timedelta(cadence)
            if pd.isna(step) or step < pd.Timedelta(seconds=1) or step.value % 1_000_000_000:
                raise ValueError("invalid duration")
            step = step.to_pytimedelta()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("cadence must be a positive finite duration in whole seconds, e.g. '12min'") from exc
        timestamps = []
        timestamp = start
        while timestamp <= end:
            timestamps.append(timestamp)
            if end - timestamp < step:
                break
            timestamp += step
        include_aia = kwargs.pop("include_aia", self.include_aia)
        if include_aia is None:
            include_aia = self.include_aia
        paths = {} if save_dir is None else {
            stamp: Path(save_dir).expanduser() / f"observation_{stamp.strftime('%Y%m%dT%H%M%S.%fZ')}.npz"
            for stamp in timestamps
        }
        cached = {stamp for stamp, path in paths.items() if path.exists()} if not overwrite else set()
        missing = [stamp for stamp in timestamps if stamp not in cached]
        if not missing:
            samples = (sample for sample in ())
        elif include_aia:
            samples = self.aia_processor._iter_range(missing, on_error=on_error, progress=progress)
        else:
            samples = (
                (stamp, None) for stamp in tqdm(
                    missing, desc="Fetching observations", unit="observation", disable=not progress,
                )
            )
        grid = tqdm(timestamps, desc="Loading saved observations", unit="observation", disable=not progress) if not missing else timestamps
        pending = None
        try:
            for timestamp in grid:
                #First yield any cached observation, then fetch/preprocess the next missing one.
                if timestamp in cached:
                    observation = Observation.load(paths[timestamp])
                    if observation.metadata.timestamp != timestamp:
                        raise ValueError(f"Cached observation timestamp does not match {paths[timestamp]}; use overwrite=True")
                    if include_aia and observation.data is None:
                        raise ValueError(f"Cached observation has no AIA data: {paths[timestamp]}; use overwrite=True")
                    include_harps = kwargs.get("include_harps")
                    include_sxr = kwargs.get("include_sxr")
                    yield Observation(
                        metadata=observation.metadata,
                        data=observation.data if include_aia else None,
                        harp_regions=observation.harp_regions if (
                            self.include_harps if include_harps is None else include_harps
                        ) else None,
                        sxr=observation.sxr if (
                            self.include_sxr if include_sxr is None else include_sxr
                        ) else None,
                    )
                    continue
                #Set pending to the next sample if it is not already set, then check if it matches the current timestamp.
                if pending is None:
                    pending = next(samples, None)
                if pending is None or pending[0] != timestamp:
                    continue  # A missing AIA sample was skipped; retain lookahead.
                #Extract the preprocessed stack and fetch the supplemental observations for this timestamp.
                _, stack = pending
                pending = None
                #Create observation from the preprocessed stack and supplemental sources. If any of those fail, skip this timestamp if on_error='skip'.
                try:
                    auxiliary = self.fetch_observation(timestamp, include_aia=False, **kwargs)
                    observation = Observation(
                        metadata=auxiliary.metadata, data=stack,
                        harp_regions=auxiliary.harp_regions, sxr=auxiliary.sxr,
                    )
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    logger.warning("Skipping observation at %s: %s", timestamp.isoformat(), exc)
                    continue
                if save_dir is not None:
                    observation.save(paths[timestamp], overwrite=overwrite)
                yield observation
        finally:
            samples.close()
            if not missing:
                grid.close()


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
        upper flux bounds in W/m². ``None`` for deterministic checkpoints.
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

    def __post_init__(self):
        object.__setattr__(
            self, "harp_regions", _harp_region_fluxes(
                self.flux_map, self.harp_regions, self.patch_uncertainty_map,
            ),
        )


def _make_aia_snapshot(image, prediction, wavelength):
    """Copy one channel into a visualization-only Observation with its provenance."""
    metadata = prediction.metadata
    if metadata is None:
        raise ValueError("Every series prediction needs timestamp metadata")
    return Observation(
        data=np.array(image, copy=True)[np.newaxis],
        metadata=ObservationMetadata(
            timestamp=metadata.timestamp, image_shape=(1, *image.shape),
            wavelengths=(wavelength,), padding_factor=metadata.padding_factor,
            harp_bbox_expansion=metadata.harp_bbox_expansion,
        ),
        harp_regions=prediction.harp_regions, sxr=prediction.sxr,
    )


def _positive_flux(values):
    """Return log-plottable values, leaving nonpositive/unknown samples as gaps."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


@dataclass(frozen=True)
class PredictionSeries:
    """Timestamped predictions with full-disk and per-HARP light curves.

    Times are requested UTC times, not individual channel exposure times.
    Duplicate requested timestamps are rejected. Missing HARP detections
    become NaN, never zero; skipped AIA samples are absent from the series.
    Regional fluxes follow Prediction's positive patch-center box sums,
    including overlapping boxes and additive variance attribution.

    Parameters
    ----------
    predictions : iterable of Prediction
        Outputs with non-None ObservationMetadata and unique timestamps.
        Stored as a tuple sorted by UTC time.
    snapshot : Observation or None, optional
        Default one-channel snapshot shaped (1, H, W), with timestamp matching
        a prediction. This visualization container is not a seven-channel model
        input. Usually populated by Predictor.predict_series.
    """

    predictions: tuple[Prediction, ...]
    snapshot: Observation | None = field(default=None, kw_only=True)

    @staticmethod
    def _timestamp(prediction):
        if prediction.metadata is None:
            raise ValueError("Every series prediction needs timestamp metadata")
        return prediction.metadata.timestamp

    def __post_init__(self):
        #Initialize and validate the series.
        predictions = tuple(self.predictions)
        predictions = tuple(sorted(predictions, key=self._timestamp))
        times = [self._timestamp(p) for p in predictions]
        if len(set(times)) != len(times):
            raise ValueError("Series predictions must have unique timestamps")
        if self.snapshot is not None:
            snapshot = self.snapshot
            if snapshot.data is None or snapshot.data.shape[0] != 1:
                raise ValueError("Series snapshot must contain one processed AIA channel")
            if snapshot.metadata.timestamp not in times:
                raise ValueError("Snapshot timestamp must match a series prediction")
        object.__setattr__(self, "predictions", predictions)

    def to_dataframe(self):
        """Return full-disk predicted and measured XRS-B flux (W/m²)."""
        rows = [{
            "timestamp": self._timestamp(p), "flux": p.flux,
            "observed_xrsb_flux": p.sxr.get("xrsb_flux") if p.sxr else None,
        } for p in self.predictions]
        frame = pd.DataFrame(rows, columns=["timestamp", "flux", "observed_xrsb_flux"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.set_index("timestamp")

    def harp_light_curves(self):
        """Return ``{harpnum: DataFrame}`` indexed by UTC requested time.

        Columns ``flux_sum`` and ``flux_sum_uncertainty`` use W/m². Each
        frame covers every successful prediction; an absent HARP or unknown
        uncertainty is NaN. No query, interpolation, or zero filling occurs.
        """
        #Get the index of the predictions (based on their timestamps)
        index = self.to_dataframe().index
        #For each prediction, get the harpnum from its harp_regions (if any), and collect all unique harpnums across the series.
        ids = sorted({r["harpnum"] for p in self.predictions for r in p.harp_regions or []})

        #For the available harps, loop through each num and collect the flux_sum and flux_sum_uncertainty for each prediction, creating a DataFrame for each harpnum.
        curves = {}
        for harpnum in ids:
            rows = []
            for prediction in self.predictions:
                region = next((r for r in prediction.harp_regions or [] if r["harpnum"] == harpnum), {})
                rows.append([region.get("flux_sum", np.nan), region.get("flux_sum_uncertainty", np.nan)])
            curves[harpnum] = pd.DataFrame(
                rows, index=index, columns=["flux_sum", "flux_sum_uncertainty"], dtype=float,
            )
        return curves

    @staticmethod
    def _select_harps(curves, harpnums, min_peak_flux):
        """Validate requested IDs and keep only curves reaching the peak threshold."""
        try:
            if not np.isscalar(min_peak_flux):
                raise ValueError("not scalar")
            threshold = float(min_peak_flux)
        except (ValueError, TypeError) as exc:
            raise ValueError("min_peak_flux must be finite and nonnegative (W/m²)") from exc
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("min_peak_flux must be finite and nonnegative (W/m²)")
        requested = list(curves) if harpnums is None else list(harpnums)
        unknown = set(requested) - curves.keys()
        if unknown:
            raise ValueError(f"No predictions for HARP numbers {sorted(unknown)}")
        selected = []
        for harpnum in requested:
            flux = _positive_flux(curves[harpnum]["flux_sum"])
            if np.isfinite(flux).any() and np.nanmax(flux) >= threshold:
                selected.append(harpnum)
        return selected

    @staticmethod
    def _plot_axes(ax, snapshot_ax, aia_ax, show_snapshot):
        """Create a curve-only or three-panel layout, or validate supplied axes."""
        import matplotlib.pyplot as plt

        if not show_snapshot:
            if ax is None:
                _, ax = plt.subplots(layout="constrained")
            return ax, snapshot_ax, aia_ax
        if ax is not None and (snapshot_ax is None or aia_ax is None):
            raise ValueError("Supply snapshot_ax and aia_ax with ax, or omit ax to create all three panels")
        if snapshot_ax is not None or aia_ax is not None:
            if (ax is None or snapshot_ax is None or aia_ax is None
                    or len({id(ax), id(snapshot_ax), id(aia_ax)}) != 3
                    or snapshot_ax.figure is not ax.figure or aia_ax.figure is not ax.figure):
                raise ValueError("ax, snapshot_ax, and aia_ax must be different Axes in the same figure")
        if ax is None:
            _, (ax, snapshot_ax, aia_ax) = plt.subplots(
                1, 3, figsize=(24, 8), gridspec_kw={"width_ratios": [1.5, 1, 1]},
                layout="constrained",
            )
        return ax, snapshot_ax, aia_ax

    def plot(self, *, harpnums=None, ax=None, include_full_disk=False, include_sxr=False,
             show_snapshot=False, snapshot_ax=None, aia_ax=None,
             show_uncertainty=True, min_peak_flux=0.0, legend="auto"):
        """Plot selected HARP curves (all by default), with optional totals.

        Missing HARP rows break lines. ``show_uncertainty=True`` shades
        regional flux +/- one flux_sum_uncertainty and enables snapshot
        uncertainty labels, only where available. Deterministic predictions
        have no bands or +/- labels. These Gaussian variance attributions
        are not calibrated regional errors. The flux axis is logarithmic:
        nonpositive/nonfinite flux samples break lines, and bands whose lower
        bound is nonpositive are omitted rather than clipped onto the axis.
        ``min_peak_flux`` selects HARPs whose maximum finite positive flux
        over the entire series is at least this threshold, in W/m². Regions
        with no positive samples are hidden. It filters snapshot HARP labels
        too, without changing the returned tables or predictions.
        ``legend='auto'`` shows a compact legend below the curves when a
        snapshot is visible (HARP IDs remain on the flux map), otherwise a
        full legend below the curves. Use 'full', 'compact', or 'none' to
        override. HARP colors remain stable when filtering.
        Returns a matplotlib Axes. Use ``harpnums=[]`` for totals only.
        ``show_snapshot=True`` adds the selected prediction's standard
        ``plot_flux_map`` panel, then its retained AIA image on the far right.
        Flux-map HARP overlays share the curve colors; AIA has no overlays.
        Create it with ``predict_series(..., snapshot_time=...)`` first.
        Only selected HARPs present at the snapshot time are outlined. A
        dashed line marks that requested UTC time in the curves. If ax is
        supplied, snapshot_ax (flux map) and aia_ax must be distinct Axes in
        the same figure; otherwise all three are created automatically.
        The returned Axes
        is always the light-curve panel. Disabling shows only the curves.
        The snapshot is the observation nearest the target supplied to predict_series,
        with earlier observations winning ties.

        Parameters
        ----------
        harpnums : iterable of int or None, optional
            HARP identifiers to display, e.g. [1234, 5678]. None considers all
            regions; [] displays only any enabled comparison curves. A single
            integer must be wrapped in a list. Unknown IDs raise ValueError.
        ax : matplotlib.axes.Axes or None, optional
            Curve axes. None creates a figure; with snapshots, custom axes require
            all three of ax, snapshot_ax, and aia_ax in the same figure.
        include_full_disk, include_sxr : bool, default False
            Add the full-disk prediction and measured GOES XRS-B comparison,
            respectively. Neither comparison is affected by min_peak_flux.
        show_snapshot : bool, default False
            Add the selected flux map and matching plain AIA image panels.
        snapshot_ax, aia_ax : matplotlib.axes.Axes or None, optional
            Flux-map and AIA axes, distinct from ax and each other. Omit all three
            axes for automatic layout.
        show_uncertainty : bool, default True
            Show available regional uncertainty shading and flux-map +/- labels.
            Deterministic outputs never receive artificial uncertainty values.
        min_peak_flux : float, default 0.0
            Finite nonnegative HARP peak threshold in W/m², e.g. 1e-7. Tests the
            maximum positive predicted regional flux across the whole series,
            not just the snapshot. Filters visuals only; returned tables stay complete.
        legend : {"auto", "full", "compact", "none"}, default "auto"
            Legend below the curve axes. Auto uses compact with snapshots, full
            otherwise. Compact shows a region count and comparison/marker keys.

        Returns
        -------
        matplotlib.axes.Axes
            Light-curve axes. Its figure contains the optional snapshot panels.

        Examples
        --------
        >>> ax = series.plot(show_snapshot=True,
        ...                  min_peak_flux=1e-7, legend="compact")
        """
        if legend not in {"auto", "full", "compact", "none"}:
            raise ValueError("legend must be 'auto', 'full', 'compact', or 'none'")
        curves = self.harp_light_curves()
        selected = self._select_harps(curves, harpnums, min_peak_flux)
        snapshot = self.snapshot if show_snapshot else None
        if show_snapshot and snapshot is None:
            raise ValueError("No AIA snapshot retained; use predict_series(..., snapshot_time=...)")
        #Creates subplots for the curves, flux map, and AIA image if show_snapshot is True. If ax is supplied, snapshot_ax and aia_ax must also be supplied and distinct. 
        ax, snapshot_ax, aia_ax = self._plot_axes(ax, snapshot_ax, aia_ax, show_snapshot)

        # Assign colors before filtering so each HARP keeps its identity.
        colors = {harpnum: _harp_rank_color(rank) for rank, harpnum in enumerate(curves)}
        self._plot_curves(ax, curves, selected, colors, include_full_disk, include_sxr, show_uncertainty)
        if snapshot is not None:
            self._plot_snapshot_panels(ax, snapshot_ax, aia_ax, snapshot, selected, colors, show_uncertainty)
        self._plot_legend(ax, selected, legend, show_snapshot)
        return ax

    def _plot_curves(self, ax, curves, selected, colors, include_full_disk, include_sxr, show_uncertainty):
        """Draw log-safe regional/comparison curves, uncertainty, and UTC axes."""
        has_uncertainty = False
        ax.set_yscale("log")
        for harpnum in selected:
            curve = curves[harpnum]
            flux = _positive_flux(curve["flux_sum"])
            ax.plot(curve.index, flux, color=colors[harpnum], marker=".", markersize=2.5, label=f"HARP {harpnum}")
            if show_uncertainty:
                sigma = curve["flux_sum_uncertainty"].to_numpy(dtype=float)
                lower, upper = flux - sigma, flux + sigma
                valid = np.isfinite(flux) & np.isfinite(sigma) & (sigma >= 0) & (lower > 0) & np.isfinite(upper)
                if valid.any():
                    ax.fill_between(
                        curve.index, np.where(valid, lower, np.nan), np.where(valid, upper, np.nan),
                        where=valid, color=colors[harpnum], alpha=0.18, linewidth=0,
                    )
                    has_uncertainty = True
        totals = self.to_dataframe()
        if include_full_disk:
            ax.plot(totals.index, _positive_flux(totals["flux"]), color="black", marker=".", markersize=2.5, label="FOXES full disk")
        if include_sxr:
            ax.plot(totals.index, _positive_flux(totals["observed_xrsb_flux"]), color="0.5", linestyle="--", marker=".", markersize=2.5, label="Observed GOES XRS-B")
        ax.set(xlabel="Requested time (UTC)", ylabel="SXR flux (W/m²)")
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, date2num
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator, tz=timezone.utc))
        #Setting limits for the x-axis based on the first and last timestamps in the totals DataFrame. If the start and end are the same, it expands the range by 30 seconds on each side to avoid a zero-width axis.
        if len(totals.index):
            start, end = (float(date2num(stamp.to_pydatetime())) for stamp in (totals.index[0], totals.index[-1]))
            if start == end:
                start, end = start - 30 / 86400, end + 30 / 86400
            ax.set_xlim(start, end)
        if has_uncertainty:
            ax.set_title(r"HARP light curves (shading: $\pm1\sigma$ variance attribution)")

    def _plot_snapshot_panels(self, ax, snapshot_ax, aia_ax, snapshot, selected, colors, show_uncertainty):
        """Reuse the standard flux map beside its matching plain AIA image."""
        from matplotlib.dates import date2num

        assert snapshot is not None and snapshot.data is not None
        assert snapshot_ax is not None and aia_ax is not None
        prediction = next(p for p in self.predictions if self._timestamp(p) == snapshot.metadata.timestamp)
        regions = [r for r in prediction.harp_regions or []
                   if r["harpnum"] in selected and np.isfinite(r["flux_sum"]) and r["flux_sum"] > 0]
        aia_ax.set_title(
            f"AIA {snapshot.metadata.wavelengths[0]} Å\n"
            f"{snapshot.metadata.timestamp:%Y-%m-%d %H:%M:%S} UTC"
        )
        ax.axvline(float(date2num(_utc_datetime(snapshot.metadata.timestamp))), color="0.5", linestyle=":", label="AIA snapshot")
        plot_flux_map(
            prediction, ax=snapshot_ax,
            show_harp_boxes=True, harp_regions=regions, harp_colors=colors,
            min_harp_flux=0, show_harp_uncertainty=show_uncertainty, harp_label_fontsize=8,
        )
        image = snapshot.data[0]
        low, high = np.percentile(image, [1, 99])
        if low == high:
            low, high = low - 0.5, high + 0.5
        aia_ax.imshow(image, origin="lower", cmap="gray", vmin=low, vmax=high)
        aia_ax.set(xlabel="X (AIA pixels)", ylabel="Y (AIA pixels)")

    @staticmethod
    def _plot_legend(ax, selected, legend, show_snapshot):
        """Put the full/compact legend below the curve axes, or omit it."""
        mode = ("compact" if show_snapshot else "full") if legend == "auto" else legend
        if mode != "none":
            handles, labels = ax.get_legend_handles_labels()
            if mode == "compact":
                from matplotlib.lines import Line2D
                entries = [(handle, label) for handle, label in zip(handles, labels) if not label.startswith("HARP ")]
                if selected:
                    caption = f"HARP regions ({len(selected)})" + ("; IDs on flux map" if show_snapshot else "")
                    entries.insert(0, (Line2D([], [], color="0.4"), caption))
                handles, labels = ([entry[0] for entry in entries], [entry[1] for entry in entries])
            if handles:
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                          ncol=min(3, len(handles)), frameon=False, fontsize=8)



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
        """Load the checkpoint and its optional input normalization.

        Parameters
        ----------
        checkpoint_path : str or pathlib.Path
            Existing full .ckpt/model file, not a Git LFS pointer; ~ is expanded.
        device : {"auto", "cpu", "cuda", "mps"}, default "auto"
            Compute device; auto selects an available accelerator.
        aia_norm_path : str or pathlib.Path or None, optional
            Matching .npz/.csv AIA normalization artifact. Supply only if input
            arrays have not already received that normalization.
        """
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

    def predict_series(self, observations, *, snapshot_time=None, snapshot_wavelength=193):
        """Predict observations, optionally keeping the AIA image nearest a target time.

        The iterable is consumed once, reusing this model. With a snapshot target,
        each observation replaces the retained image only if it is closer to that
        time (earlier wins ties). Only one channel image is kept in memory.
        No temporary image files are written. Prediction failures raise.

        Parameters
        ----------
        observations : iterable of Observation
            List, tuple, or generator such as processor.iter_observations(...) or
            load_observations("saved_aia"). Each item needs a finite seven-channel
            processed array matching the checkpoint. Wrap plain arrays in an
            Observation first.
        snapshot_time : datetime.datetime, datetime.date, numpy.datetime64, str, or None
            Target time for the optional snapshot, e.g. "2024-03-23 01:40".
            Accepts all parse_timestamp formats; naive values mean UTC.
            None retains no image. To display the query midpoint, pass
            start + (end - start) / 2 using parsed datetime values.
        snapshot_wavelength : int, default 193
            Angstrom channel to retain, e.g. 171 or 193, not a channel index.
            Must occur in each observation's metadata.wavelengths when a
            snapshot_time is supplied.

        Returns
        -------
        PredictionSeries
            Timestamped predictions and an optional one-channel snapshot for
            series.plot(show_snapshot=True). Full input stacks are not retained.
        """
        target = _utc_datetime(snapshot_time) if snapshot_time is not None else None
        predictions = []
        snapshot = None
        best_distance = timedelta.max

        for observation in observations:
            if not isinstance(observation, Observation):
                raise TypeError("predict_series expects Observation instances")
            if target is not None and snapshot_wavelength not in observation.metadata.wavelengths:
                raise ValueError(f"Snapshot wavelength {snapshot_wavelength} Å is absent from the observation")
            prediction = self.predict(observation)
            predictions.append(prediction)
            if target is None:
                continue

            # Keep this image only if it is closer, or earlier on an equal distance.
            timestamp = _utc_datetime(observation.metadata.timestamp)
            distance = abs(timestamp - target)
            if (snapshot is None or distance < best_distance
                    or (distance == best_distance and timestamp < _utc_datetime(snapshot.metadata.timestamp))):
                if observation.data is None:
                    raise ValueError("AIA data is required to retain a snapshot")
                channel = observation.metadata.wavelengths.index(snapshot_wavelength)
                snapshot = _make_aia_snapshot(observation.data[channel], prediction, snapshot_wavelength)
                best_distance = distance

        return PredictionSeries(tuple(predictions), snapshot=snapshot)

    def predict(self, image, *, timestamp=None):
        """Predict global SXR flux and patch contributions from one AIA stack.

        Parameters
        ----------
        image : Observation or array-like
            Finite channel-first processed AIA stack in 94, 131, 171, 193,
            211, 304, 335 Å order. Shape is ``(7, H, W)`` with ``H`` and
            ``W`` determined by the checkpoint's patch size and grid.
            Observation metadata is copied into the returned prediction.
        timestamp : datetime.datetime, datetime.date, numpy.datetime64, str, or None
            Time for a plain array; all parse_timestamp formats work, e.g.
            "2023/08/01 12:00" or "2023-08-01T12:00:00Z". Naive values
            mean UTC; None leaves an array prediction without time metadata. Its crop is assumed to use
            the standard AIA channel order and 0.1 padding. For other crops,
            supply an Observation with explicit ObservationMetadata.

        Returns
        -------
        Prediction
            Full-disk flux, patch flux map, and optional Gaussian uncertainty.

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
        with torch.inference_mode():
            outputs = self.model(tensor, return_attention=False)
            if getattr(self.base_model, "predicts_uncertainty", False):
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
            harp_regions=image.harp_regions if isinstance(image, Observation) else None,
            sxr=image.sxr if isinstance(image, Observation) else None,
        )


def load_model(checkpoint_path, *, device="auto", aia_norm_path=None):
    """Load a checkpoint for repeated calls to :func:`predict`.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Existing full checkpoint/model file, e.g. "models/ComplexModel.ckpt".
        Git LFS pointer text files are not model checkpoints. ~ is expanded.
    device : {"auto", "cpu", "cuda", "mps"}, default "auto"
        Inference device; auto selects an available accelerator.
    aia_norm_path : str or pathlib.Path or None, optional
        Checkpoint-matched .npz/.csv AIA normalization artifact. Apply it once:
        omit here if preprocessing already applied the same normalization.

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
    timestamp : datetime.datetime, datetime.date, numpy.datetime64, str, or None
        Array input time accepted by parse_timestamp, e.g.
        "2023/08/01 12:00"; naive values mean UTC. None omits array
        time metadata. Observation inputs carry their own
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
                  harp_label_max_shift=80, aia_image=None, harp_colors=None):
    """Plot patch flux contributions with a logarithmic color scale.

    For a Prediction without an AIA background, the title shows its predicted
    FOXES flux with the model's full-disk uncertainty when available, and the measured SXR-B
    satellite mean when available, all in W/m². The uncertainty is the
    square root of summed patch variances, not a calibrated error bar.
    Missing values are omitted. Plain map arrays do not add a flux title.

    Parameters
    ----------
    prediction : Prediction or array-like
        A FOXES result or a two-dimensional patch flux map in W/m².
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure and axes are created if omitted.
    cmap : str or matplotlib.colors.Colormap, default "inferno"
        Colormap name such as "inferno" or "gray", or a Colormap instance.
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
        For manually supplied boxes, each dict needs HARPNUM or harpnum,
        plus top_pct, left_pct, width_pct, and height_pct, in percent of
        the input crop measured from the top-left display corner.
        Defaults to the prediction's regions. If neither has been fetched
        and ``show_harp_boxes`` is true, regions are queried from JSOC. An
        empty list suppresses the query, including one carried by a prediction.
    timestamp : datetime.datetime, datetime.date, numpy.datetime64, str, or None
        Complete AIA time, e.g. "2023/08/01 12:00" or an aware
        datetime; naive values mean UTC. Accepts the same formats as
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
    aia_image : array-like or None, optional
        Display a processed 2D AIA channel instead of the patch flux map,
        reusing the same HARP box/label placement. The image must cover the
        exact crop used for prediction. Flux labels still sum predicted
        patch contributions, not AIA intensity. Uses a linear 1–99 percentile
        stretch (or vmin/vmax), pixel axes, and no flux colorbar. Pass
        ``cmap='gray'`` for a neutral background. No downloading occurs.
    harp_colors : dict[int, color] or None, optional
        Override outline/label colors by HARP number, for matching other
        plots. Unspecified regions retain descending-flux-rank colors.
        Examples: {1234: "tab:blue"}, {1234: "#56B4E9"}, or
        {1234: (0.2, 0.5, 0.8)} with RGB components in [0, 1].

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the map. Flux maps also have a flux colorbar.

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
    if aia_image is None:
        low, high = _log_limits(positive, vmin, vmax)
    else:
        aia_image = np.asarray(aia_image)
        if aia_image.ndim != 2 or not np.isfinite(aia_image).all() or aia_image.size == 0:
            raise ValueError("aia_image must be a finite nonempty 2D array of the prediction crop")
        percentiles = np.percentile(aia_image, [1, 99])
        low = float(percentiles[0]) if vmin is None else float(vmin)
        high = float(percentiles[1]) if vmax is None else float(vmax)
        if low == high and vmin is None and vmax is None:
            low, high = low - 0.5, high + 0.5
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("AIA color limits must be finite with vmin < vmax")
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
    if aia_image is None:
        artist = ax.imshow(
            np.ma.masked_less_equal(flux_map, 0), origin="lower", cmap=cmap,
            norm=LogNorm(vmin=low, vmax=high),
        )
        ax.set(xlabel="X (Patches)", ylabel="Y (Patches)")
        ax.figure.colorbar(artist, ax=ax, label="Flux contribution (W m$^{-2}$)")
    else:
        ax.imshow(aia_image, origin="lower", cmap=cmap, vmin=low, vmax=high)
        ax.set(xlabel="X (AIA pixels)", ylabel="Y (AIA pixels)")
    if isinstance(prediction, Prediction) and aia_image is None:
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
    if regions:
        rows, cols = flux_map.shape if aia_image is None else aia_image.shape
        colors = [(harp_colors or {}).get(region["harpnum"], _harp_rank_color(rank))
                  for rank, region in enumerate(regions)]
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
    cmap : str or matplotlib.colors.Colormap, default "magma"
        Colormap name such as "magma", or a Colormap instance.
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

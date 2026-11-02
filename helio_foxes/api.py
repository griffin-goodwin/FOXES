"""Fetch AIA, HARP, and GOES SXR observations and predict solar X-ray flux.

The API is a short pipeline, and this module is laid out in the same order:

1. Get data. :class:`ObservationFetcher` downloads and processes AIA, HARP,
   and GOES data and returns an :class:`Observation`. Saved observations come
   back with :func:`load_observations`, and your own arrays go in through
   :meth:`Observation.from_array`.
2. Load a model. Create a :class:`FOXESModel` from a checkpoint. Do this
   once and reuse it.
3. Predict. :meth:`FOXESModel.predict` turns an :class:`Observation` into a
   :class:`Prediction`, and :meth:`FOXESModel.predict_time_series` turns
   many into a :class:`PredictionTimeSeries`.
4. Look at the results. :class:`Prediction` holds the flux, flux map, and
   HARP fluxes and plots them; :class:`PredictionTimeSeries` gives light
   curves and plots those.

:class:`AIAProcessor`, :class:`HARPProcessor`, and :class:`SXRProcessor` are
the lower-level pieces the fetcher is built from.
"""

from __future__ import annotations

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

_CHECKPOINT_URL = "https://media.githubusercontent.com/media/griffin-goodwin/FOXES/main/model_assets/"
# Names FOXESModel accepts in place of a path, mapped to their location under model_assets/.
PRETRAINED_CHECKPOINTS = {
    "uncertainty": "UncertaintyModel.ckpt",
    "inverted-attention-mask": "LegacyModels/inverted-attention-mask.ckpt",
    "localized-attention-mask": "LegacyModels/localized-attention-mask.ckpt",
    "no-attention-mask": "LegacyModels/no-attention-mask.ckpt",
}


class ObservationFetcher:
    """Fetch everything you need for a prediction at a given time.

    This is the usual starting point. :meth:`fetch` returns an
    :class:`Observation` and :meth:`fetch_range` yields one per time; pass
    them to :class:`FOXESModel` to predict. It wraps :class:`AIAProcessor`,
    :class:`HARPProcessor`, and :class:`SXRProcessor` and bundles what they
    return into an :class:`Observation`. All three sources are on by
    default. Any of them can be turned off here or for a single call.

    The sources fail differently. If the AIA download or processing fails,
    you get an error, since there's nothing to predict on. HARP and GOES are
    extras: if they fail you get an empty region list or ``sxr=None`` and
    the observation still comes back. A source you turned off shows up as
    ``None`` (``data=None`` for AIA, ``harp_regions=None`` for HARPs).
    """

    def __init__(self, *, email=None,
                 include_aia=True, include_harps=True, include_sxr=True,
                 resolution=512, aia_tolerance=120, n_workers=4,
                 harp_bbox_expansion=0.0,
                 sxr_tolerance=120, sxr_satellite=None):
        """Set up the data sources.

        Parameters
        ----------
        email : str or None, optional
            Your registered JSOC email, used for both AIA and HARP queries.
            Defaults to the ``JSOC_EMAIL`` environment variable.
        include_aia, include_harps, include_sxr : bool, default True
            Which sources to fetch by default. Each fetch call can override
            these.
        resolution : int, default 512
            Processed AIA image size in pixels. Should match the checkpoint.
        aia_tolerance : int, default 120
            How far, in seconds, each AIA frame can be from the requested time.
        n_workers : int, default 4
            Number of parallel AIA download threads.
        harp_bbox_expansion : float, default 0.0
            Fraction of each HARP box's width and height added on every side
            (0.1 grows each side by 10%). HARP flux sums use the expanded boxes.
        sxr_tolerance : float, default 120
            How far, in seconds, a GOES sample can be from the requested time.
        sxr_satellite : int or None, optional
            Use only this GOES satellite (e.g. ``18``). ``None`` averages all
            satellites with good data.
        """
        self.include_aia = include_aia
        self.include_harps = include_harps
        self.include_sxr = include_sxr
        self.aia_processor = AIAProcessor(
            email=email, resolution=resolution, tolerance=aia_tolerance,
            n_workers=n_workers,
        )
        self.harp_processor = HARPProcessor(
            email=email, resolution=self.aia_processor.resolution,
            padding_factor=self.aia_processor.PADDING_FACTOR,
            bbox_expansion=harp_bbox_expansion,
        )
        self.sxr_processor = SXRProcessor(tolerance=sxr_tolerance, satellite=sxr_satellite)

    def fetch(self, time, *, include_aia=None, include_harps=None,
              include_sxr=None) -> Observation:
        """Fetch one observation.

        Raw downloads are deleted before the method returns. The HARP and
        GOES data stay attached to the observation, so they show up again
        when you predict and plot.

        Parameters
        ----------
        time : datetime.datetime, datetime.date, numpy.datetime64, or str
            Any format :func:`parse_timestamp` accepts.
        include_aia, include_harps, include_sxr : bool or None, optional
            Turn a source on or off for this call only. ``None`` keeps the
            fetcher's default. You need AIA to make a prediction.

        Returns
        -------
        Observation
        """
        timestamp = _utc_datetime(time)
        if include_aia is None:
            include_aia = self.include_aia
        if include_harps is None:
            include_harps = self.include_harps
        if include_sxr is None:
            include_sxr = self.include_sxr
        stack = self.aia_processor.fetch(timestamp) if include_aia else None
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
        regions = self.harp_processor.fetch(timestamp) if include_harps else None
        sxr = self.sxr_processor.fetch(timestamp) if include_sxr else None
        return Observation(
            metadata=metadata, data=stack, harp_regions=regions, sxr=sxr,
        )

    def fetch_range(self, start, end, *, cadence="1h", on_error="raise",
                    save_dir=None, overwrite=False, progress=False,
                    include_aia=None, include_harps=None, include_sxr=None):
        """Fetch observations over a time range, one at a time.

        This is a generator, so nothing gets downloaded until you start
        looping over it (or hand it to :meth:`FOXESModel.predict_time_series`).
        Only one AIA stack is held in memory at a time, which keeps long
        ranges manageable.

        **Which times.** The range starts at ``start`` and steps forward by
        ``cadence``. ``end`` is included only if it falls exactly on one of
        those steps. There's no interpolation, so two requested times close
        together can end up using the same AIA exposure.

        **How it downloads.** AIA frames for the whole range are found with
        a single JSOC query, then downloaded. HARP and GOES are still queried
        separately for each time. Very long ranges can produce large JSOC
        responses, so it's worth splitting them into chunks.

        **Saving and reusing.** With ``save_dir`` set, each observation is
        saved with :meth:`Observation.save` before it's yielded. On a later
        run, any time that already has a file is loaded from disk, with no
        downloads or queries, and only the missing times are fetched. Saved
        files are reused exactly as they are, so if you change preprocessing
        or source settings, pass ``overwrite=True`` to fetch everything again.
        Reload a saved folder without the fetcher using
        :func:`load_observations`.

        **Errors.** By default an AIA failure stops the iterator. With
        ``on_error="skip"``, the failure is logged with its time and the
        iterator carries on. HARP and GOES failures never stop it (see
        :meth:`fetch`). Problems with the initial JSOC query, or
        with reading or writing saved files, always raise.

        Parameters
        ----------
        start, end : datetime.datetime, datetime.date, numpy.datetime64, or str
            First and last times, in any format :func:`parse_timestamp`
            accepts. ``end`` can't be before ``start``.
        cadence : str, datetime.timedelta, pandas.Timedelta, int, or float, default "1h"
            Time between observations, as a whole number of seconds. Use a
            pandas-style string like ``"12min"``, a timedelta, or a plain
            number of seconds (so ``720`` is 12 minutes).
        on_error : {"raise", "skip"}, default "raise"
            What to do when a single time fails to download or process.
        save_dir : str or pathlib.Path or None, optional
            Folder to save observations in and reuse them from.
        overwrite : bool, default False
            Fetch every time again and replace any saved files.
        progress : bool, default False
            Show a tqdm progress bar for the JSOC query and downloads, or for
            loading files if everything is already saved. If AIA is turned
            off, it tracks the HARP and GOES fetches instead.
        include_aia, include_harps, include_sxr : bool or None, optional
            Turn a source on or off for this range only, as in :meth:`fetch`.
            ``None`` keeps the fetcher's default.

        Yields
        ------
        Observation
            One per successful time, in time order.

        Examples
        --------
        >>> observations = fetcher.fetch_range(
        ...     "2023-08-01 00:00", "2023-08-01 01:00", cadence="12min",
        ...     save_dir="saved_observations", progress=True,
        ... )
        >>> series = model.predict_time_series(observations)
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
        include_aia = self.include_aia if include_aia is None else include_aia
        include_harps = self.include_harps if include_harps is None else include_harps
        include_sxr = self.include_sxr if include_sxr is None else include_sxr
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
                    yield Observation(
                        metadata=observation.metadata,
                        data=observation.data if include_aia else None,
                        harp_regions=observation.harp_regions if include_harps else None,
                        sxr=observation.sxr if include_sxr else None,
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
                    auxiliary = self.fetch(
                        timestamp, include_aia=False,
                        include_harps=include_harps, include_sxr=include_sxr,
                    )
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
class Observation:
    """One processed AIA stack, with its metadata and any HARP or GOES data.

    You'll usually get these from :class:`ObservationFetcher`, or build
    one from your own array with :meth:`from_array`. Pass it to
    :meth:`FOXESModel.predict` to get a :class:`Prediction`.

    Parameters
    ----------
    metadata : ObservationMetadata
        Time, channel order, and crop geometry for ``data``.
    data : array-like or None
        Channel-first ``(C, H, W)`` array matching ``metadata.image_shape``.
        ``None`` if AIA fetching was turned off. :meth:`from_array` checks
        the values and converts them to float32; this constructor doesn't.
    harp_regions : list of dict or None, optional
        Active-region boxes. Each dict needs a ``HARPNUM`` (or ``harpnum``)
        and ``top_pct``, ``left_pct``, ``width_pct``, and ``height_pct``,
        given as percentages of this crop measured from its top-left corner.
        For example ``{"HARPNUM": 1234, "top_pct": 10, "left_pct": 20,
        "width_pct": 15, "height_pct": 12}``. ``None`` means HARPs were never
        requested. An empty list means they were requested but nothing usable
        came back, which includes failed queries. Prediction won't query
        HARPs for you.
    sxr : dict or None, optional
        Measured GOES flux from :meth:`SXRProcessor.fetch`, with
        ``xrsa_flux`` and ``xrsb_flux`` in W/m². It's only there for
        comparison; the model never sees it. ``None`` if SXR was turned off
        or no data was available.
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
        """Wrap an AIA stack you've already preprocessed.

        The array is used as is. Nothing is downloaded, and there's no
        calibration, cropping, resizing, or normalization, so the stack needs
        to match the training preprocessing already. The model checks shape and channel order when you run it.

        Parameters
        ----------
        data : array-like
            Finite numeric ``(C, H, W)`` stack, usually ``(7, 512, 512)``.
            A NumPy array or nested lists both work. Channels-last arrays
            aren't transposed for you. The observation stores a float32 copy.
        timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str
            Required. Any format :func:`parse_timestamp` accepts.
        wavelengths : sequence of int, optional
            Wavelength of each channel in Ångströms. Prediction needs the
            default ``(94, 131, 171, 193, 211, 304, 335)``.
        padding_factor : float, default 0.1
            Padding past the solar limb used when the stack was cropped.
        harp_regions : list of dict or None, optional
            Active-region boxes in percent of this crop; see :class:`Observation`.
        sxr : dict or None, optional
            Measured GOES flux; see :class:`Observation`.

        Returns
        -------
        Observation
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

    def save(self, path, *, overwrite=False):
        """Save the whole observation to a compressed ``.npz`` file.

        The file holds the array (if there is one) and the metadata: time,
        crop geometry, channel order, HARP regions, and SXR data. Reload it
        with :meth:`load` and it's ready to predict on, with no need to
        preprocess or normalize it again. The data is saved exactly as is,
        and pickle isn't used.

        The file is written to a temporary name first and only moved into
        place once it's complete, so an interrupted save won't leave a
        half-written archive behind.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination ending in ``.npz``. Missing parent folders are created
            and ``~`` is expanded.
        overwrite : bool, default False
            Replace an existing file. If False and the file already exists,
            raises ``FileExistsError``.

        Returns
        -------
        pathlib.Path
            Where the archive was written.
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
        """Reload an observation written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            A ``.npz`` archive from :meth:`save`. For a plain ``.npy`` array,
            use ``Observation.from_array(np.load(path), timestamp=...)``.

        Returns
        -------
        Observation
            The saved array and metadata, along with any HARP and SXR data.
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


@dataclass(frozen=True)
class ObservationMetadata:
    """Time, channel order, and crop geometry for one processed AIA stack.

    The metadata travels with an observation through prediction and
    plotting, so HARP boxes can be drawn in the right place later on.

    Parameters
    ----------
    timestamp : datetime.datetime, datetime.date, numpy.datetime64, or str
        The time you asked for, in any format :func:`~helio_foxes.parse_timestamp`
        accepts. Each channel's actual exposure time can differ slightly.
    image_shape : sequence of three positive int
        Array shape in channel-first order, e.g. ``(7, 512, 512)``. If AIA
        fetching was turned off, this is the shape the crop would have had.
    wavelengths : sequence of int, optional
        Wavelength of each channel in Ångströms, in array order. Prediction
        needs the default ``(94, 131, 171, 193, 211, 304, 335)``.
    padding_factor : float, default 0.1
        How far the crop extends past the solar limb, as a fraction of the
        solar radius. Please keep it consistent with the crop used to train the model (0.1).
    harp_bbox_expansion : float, default 0.0
        Fraction of each HARP box's width and height added on every side, so
        0.1 grows each side by 10%. Stored here so HARPs fetched later, at
        plotting time, get the same expansion.
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


def load_observations(directory, *, progress=False):
    """Load a folder of saved observations, one at a time.

    Files are read in filename order. The names that
    :meth:`ObservationFetcher.fetch_range` writes are timestamps,
    so that's chronological order. Only one file is in memory at a time,
    so you can pass the result straight to :meth:`FOXESModel.predict_time_series`::

        series = model.predict_time_series(load_observations("saved_observations"))

    Raises ``FileNotFoundError`` if the folder is missing or has no
    ``.npz`` files. A broken archive raises an error rather than being
    skipped.

    Parameters
    ----------
    directory : str or pathlib.Path
        Folder of ``.npz`` files written by :meth:`Observation.save`. ``~``
        is expanded. Every ``.npz`` file in the folder is loaded, so keep
        normalization files and other archives somewhere else.
    progress : bool, default False
        Show a tqdm progress bar.

    Yields
    ------
    Observation
    """
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Observation directory does not exist: {directory}")
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No observation .npz archives found in {directory}")
    for path in tqdm(paths, desc="Loading observations", unit="observation", disable=not progress):
        yield Observation.load(path)


class FOXESModel:
    """A FOXES model loaded from a checkpoint, ready to make predictions.

    Loading takes a while, so do it once and reuse the model::

        model = FOXESModel()  # the current uncertainty model
        result = model.predict(observation)

    Give :meth:`predict` an :class:`Observation` to get a :class:`Prediction`,
    or give :meth:`predict_time_series` many of them to get a
    :class:`PredictionTimeSeries`.

    Parameters
    ----------
    checkpoint : str or pathlib.Path, default "uncertainty"
        A released checkpoint name or a path to a checkpoint file. The names
        are ``"uncertainty"`` (the current model) and the legacy
        ``"inverted-attention-mask"``, ``"localized-attention-mask"``, and
        ``"no-attention-mask"``. A named checkpoint is downloaded the first
        time you use it and cached in ``~/.cache/helio_foxes`` (set the
        ``HELIO_FOXES_CACHE`` environment variable to use another folder).
        If a file with that name exists, the file wins. ``~`` is expanded in
        paths.
    device : {"auto", "cpu", "cuda", "mps"}, default "auto"
        Where to run the model. ``"auto"`` uses a GPU if one is available.
    legacy_correction_enabled : bool or None, default None
        Turn on the approximate correction for the old deterministic
        checkpoints: flux (full-disk and per patch) is multiplied by 0.7 for
        times before ``legacy_cutoff``. ``None`` turns it on for the legacy
        checkpoint names and off for everything else, including file paths,
        so pass ``True`` if you load a legacy checkpoint from a file. Keep it
        off for ``UncertaintyModel.ckpt`` and any model trained on the
        corrected targets. The weights and normalization themselves aren't
        changed.
    legacy_cutoff : datetime.datetime or str, default "2016-11-19"
        Predictions before this UTC time get the 0.7 factor. It's a rough
        stand-in for the change in GOES satellites, not a fitted calibration.
    """

    def __init__(self, checkpoint="uncertainty", *, device="auto",
                 legacy_correction_enabled=None, legacy_cutoff="2016-11-19"):
        """Load the checkpoint; see the class docstring."""
        from forecasting.inference import load_model_from_config
        from forecasting.calibration import LegacyCalibration

        if legacy_correction_enabled is not None and not isinstance(legacy_correction_enabled, bool):
            raise ValueError("legacy_correction_enabled must be a boolean")
        if legacy_correction_enabled is None:
            legacy_correction_enabled = PRETRAINED_CHECKPOINTS.get(str(checkpoint), "").startswith("LegacyModels/")
        self.legacy_calibration = LegacyCalibration(legacy_correction_enabled, legacy_cutoff)
        self.legacy_correction_enabled = self.legacy_calibration.enabled
        self.legacy_cutoff = self.legacy_calibration.cutoff
        path = _resolve_checkpoint(checkpoint)
        self.model = load_model_from_config({
            "data": {"checkpoint_path": str(path)}, "device": device,
            "legacy_model": legacy_correction_enabled, "legacy_cutoff": self.legacy_cutoff,
        })
        self.device = next(self.model.parameters()).device
        self.model_kwargs = dict(self.model.hparams["model_kwargs"])

    def predict_time_series(self, observations, *, snapshot_time=None, snapshot_wavelength=193):
        """Run the model over a sequence of observations.

        Observations are processed one at a time as the iterable is consumed,
        so this works well with generators like
        :meth:`ObservationFetcher.fetch_range` and
        :func:`load_observations`. The input stacks aren't kept. If any
        prediction fails, the error is raised.

        Those generators can only be looped over once. If you pass one that
        has already been used, you get an empty series and a warning. Call
        :meth:`ObservationFetcher.fetch_range` again, or turn it into a list
        first if you need the observations more than once.

        If you pass ``snapshot_time``, one AIA channel from the observation
        closest to that time is kept (the earlier one wins a tie), so
        :meth:`PredictionTimeSeries.plot_light_curves` can show it with ``show_snapshot=True``.

        Parameters
        ----------
        observations : iterable of Observation
            A list, tuple, or generator of observations. Wrap plain arrays
            with :meth:`Observation.from_array` first.
        snapshot_time : datetime.datetime, datetime.date, numpy.datetime64, str, or None
            Time to keep a snapshot image for, in any format
            :func:`parse_timestamp` accepts. ``None`` keeps no image.
        snapshot_wavelength : int, default 193
            Which AIA channel to keep, given as a wavelength in Ångströms
            (e.g. ``171``), not as an index.

        Returns
        -------
        PredictionTimeSeries
        """
        target = _utc_datetime(snapshot_time) if snapshot_time is not None else None
        predictions = []
        snapshot = None
        best_distance = timedelta.max

        for observation in observations:
            if not isinstance(observation, Observation):
                raise TypeError("predict_time_series expects Observation instances")
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
                snapshot = _make_aia_snapshot(observation.data[channel], observation, snapshot_wavelength)
                best_distance = distance

        if not predictions:
            warnings.warn(
                "predict_time_series got no observations. Generators from fetch_range "
                "and load_observations can only be used once; call them again for a "
                "new pass.", stacklevel=2,
            )
        return PredictionTimeSeries(tuple(predictions), snapshot=snapshot)

    def predict(self, observation):
        """Predict the full-disk flux and per-patch flux map for one observation.

        Parameters
        ----------
        observation : Observation
            From :class:`ObservationFetcher`, :func:`load_observations`, or
            :meth:`Observation.from_array` (for your own preprocessed arrays).
            Its AIA stack has to be ``(7, H, W)`` with channels in the order
            94, 131, 171, 193, 211, 304, 335 Å, and ``H`` and ``W`` have to
            match the checkpoint (512 × 512 for the current one). Its
            metadata and HARPs are carried over to the result. Its measured
            GOES XRS-B flux becomes the result's ``ground_truth``.

        Returns
        -------
        Prediction

        Raises
        ------
        TypeError
            If ``observation`` isn't an :class:`Observation`.
        ValueError
            If the observation has no AIA data, the shape or channel order
            doesn't match the checkpoint, or the array has non-finite values.
        """
        import torch
        from forecasting.inference import uncertainty_from_prediction

        if not isinstance(observation, Observation):
            raise TypeError(
                "predict expects an Observation; wrap a plain array with "
                "Observation.from_array(array, timestamp=...)"
            )
        if observation.data is None:
            raise ValueError("Observation has no AIA data; fetch with include_aia=True before prediction")
        metadata = observation.metadata
        array = np.asarray(observation.data)
        patch_size = int(self.model_kwargs["patch_size"])
        patch_count = int(self.model_kwargs["num_patches"])
        grid = int(np.sqrt(patch_count))
        channels = int(self.model_kwargs["num_channels"])
        expected = (channels, grid * patch_size, grid * patch_size)
        if array.shape != expected:
            raise ValueError(f"Expected processed AIA shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("AIA image contains non-finite values")
        if metadata.wavelengths != AIAProcessor.WAVELENGTHS:
            raise ValueError("Observation wavelengths must be in FOXES AIA channel order")
        scale = self.legacy_calibration.factor(metadata.timestamp)
        tensor = torch.as_tensor(np.array(array, dtype=np.float32, copy=True))
        tensor = tensor.permute(1, 2, 0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(tensor, return_attention=False)
            if getattr(self.model, "predicts_uncertainty", False):
                flux, variance, patches, patch_variance = outputs
                intervals_raw = uncertainty_from_prediction(
                    flux, variance, self.model.sxr_norm,
                )
                prediction_intervals = {
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
                prediction_intervals = None
                patch_uncertainty = None
        flux_map = patches.reshape(grid, grid).detach().cpu().numpy()
        if scale != 1.0:
            flux_map = flux_map * scale
        ground_truth = observation.sxr.get("xrsb_flux") if observation.sxr else None
        return Prediction(
            metadata=metadata,
            total_flux=float(flux.item()) * scale, flux_map=flux_map,
            prediction_intervals=prediction_intervals, flux_std_map=patch_uncertainty,
            harp_contributions=observation.harp_regions,
            ground_truth=float(ground_truth) if ground_truth is not None else None,
            legacy_correction_enabled=self.legacy_correction_enabled, flux_scale_factor=scale,
        )


@dataclass(frozen=True)
class Prediction:
    """What the model predicts for one AIA stack.

    Returned by :meth:`FOXESModel.predict`. Read the numbers from the
    attributes below, or draw them with :meth:`plot_flux_map` and
    :meth:`plot_uncertainty_map`. ``total_flux`` is the prediction and
    ``ground_truth`` is the matched measured GOES XRS-B flux. All fluxes are
    in W/m², for the 1–8 Å band.

    The Gaussian checkpoint assigns a variance contribution to each patch.
    ``flux_std_map`` contains their square roots; the squares add to the
    full-disk variance. Each HARP's ``flux_std`` is the square root of the
    summed variance contributions inside its box. These are model-assigned
    uncertainty attributions; calibration against individual regional X-ray
    measurements hasn't been established.

    Attributes
    ----------
    metadata : ObservationMetadata or None
        Metadata from the input observation, kept so HARP boxes can be drawn
        later. The input array itself isn't kept.
    total_flux : float
        Predicted full-disk soft X-ray flux.
    flux_map : numpy.ndarray
        2D map of each patch's contribution to ``total_flux``. It's on the
        model's patch grid (64 × 64 for the current checkpoint), not at
        image resolution.
    prediction_intervals : dict or None
        ``{"68": (low, high), "95": (low, high)}`` nominal full-disk
        prediction intervals in W/m². They are asymmetric because the
        Gaussian is defined in normalized log-flux space. ``None`` for
        deterministic checkpoints.
    flux_std_map : numpy.ndarray or None
        2D map of the standard-deviation contributions described above, on
        the same patch grid as ``flux_map``. ``None`` for deterministic
        checkpoints. Add variances (squared values), rather than these values.
    harp_contributions : list of dict or None
        One dict per active region. Each has ``harpnum``, the box position
        (``top_pct``, ``left_pct``, ``width_pct``, ``height_pct``), and
        ``total_flux``, the total of the positive patch contributions whose
        centers fall inside the box. Boxes can overlap, so a patch can count
        toward more than one region. ``flux_std`` is
        ``sqrt(sum(flux_std_map**2))`` over the same patches, or
        ``None`` for deterministic checkpoints. The original JSOC columns,
        including the heliographic longitude and latitude bounds, are kept
        too. ``None`` means HARPs weren't fetched; an empty list means none
        were usable.
    ground_truth : float or None
        Matched measured GOES XRS-B flux, for comparison with ``total_flux``.
        The sample may be offset from the requested time within the fetcher's
        tolerance, and may average multiple satellites. Its provenance is in
        the input observation's ``sxr`` data. ``None`` if no sample is available.
    legacy_correction_enabled : bool
        Whether legacy correction is enabled (see :class:`FOXESModel`).
        This can be ``True`` with a scale factor of 1.0 after the cutoff.
    flux_scale_factor : float
        Factor that was applied to ``total_flux`` and ``flux_map``: 0.7 for legacy
        predictions before the cutoff date, otherwise 1.0. The measured
        ``ground_truth`` values are never scaled.
    """

    metadata: ObservationMetadata | None = field(default=None, kw_only=True)
    total_flux: float
    flux_map: np.ndarray
    prediction_intervals: dict[str, tuple[float, float]] | None = None
    flux_std_map: np.ndarray | None = None
    harp_contributions: list[dict] | None = None
    ground_truth: float | None = None
    legacy_correction_enabled: bool = field(default=False, kw_only=True)
    flux_scale_factor: float = field(default=1.0, kw_only=True)

    def __post_init__(self):
        object.__setattr__(
            self, "harp_contributions", _harp_region_fluxes(
                self.flux_map, self.harp_contributions, self.flux_std_map,
            ),
        )

    def plot_flux_map(self, *, ax=None, cmap="inferno", vmin=None, vmax=None,
                      show_harp_boxes=True, harp_regions=None,
                      min_harp_flux=1e-7, harp_label_fontsize=6,
                      show_harp_uncertainty=True, harp_label_offsets=None,
                      harp_label_max_shift=80, aia_image=None, harp_colors=None):
        """Plot the per-patch flux map on a log color scale.

        The title shows the predicted full-disk flux (with its uncertainty, for
        Gaussian checkpoints) and the measured GOES XRS-B flux if it's
        available. A plot with ``aia_image`` set gets no title. Patches with
        zero or negative flux are left blank.

        **HARP boxes.** If the prediction has HARP regions, each one is
        outlined and labeled with its HARP number and summed flux (and, if
        available, ``±`` its uncertainty). Colors go by flux rank, brightest
        first. Labels start just outside a box corner, on the side facing the
        image center. If labels overlap, the other corners are tried and then
        small nudges, up to ``harp_label_max_shift``. Any label that moves
        gets a connector line back to its box. Placement is worked out once
        when you plot, so replot if you resize the figure a lot. On a crowded
        map, raise ``min_harp_flux`` or set ``harp_label_offsets`` by hand.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. A new figure is created if omitted.
        cmap : str or matplotlib.colors.Colormap, default "inferno"
            Colormap name or instance.
        vmin, vmax : float or None, optional
            Color-scale limits in W/m² (must be positive). Default to the
            smallest and largest positive values in the map.
        show_harp_boxes : bool, default True
            Draw and label the HARP boxes, as described above. Nothing is
            drawn if there are no regions; plotting never queries JSOC.
        harp_regions : list of dict or None, optional
            Regions to draw instead of the prediction's own, for example from
            :meth:`HARPProcessor.fetch`. Hand-made boxes need ``HARPNUM``
            (or ``harpnum``) and ``top_pct``, ``left_pct``, ``width_pct``,
            ``height_pct`` in percent of the crop, from its top-left corner.
        min_harp_flux : float, default 1e-7
            Hide regions whose summed flux is below this, in W/m². The
            prediction itself is unchanged.
        harp_label_fontsize : float, default 6
            Label font size in points.
        show_harp_uncertainty : bool, default True
            Add the ``±`` uncertainty line to labels when the prediction has
            patch uncertainties.
        harp_label_offsets : dict or None, optional
            ``{harpnum: (dx, dy)}`` starting offsets in points from the chosen
            box corner; positive ``dx`` moves right and positive ``dy`` moves up.
            Labels placed this way always get a connector.
        harp_label_max_shift : float, default 80
            How far, in points, a label can be nudged automatically. Set it to
            0 to keep every label at its starting corner. Nudged labels stay
            outside their box and inside the map. A label that can't find a
            clear spot within this distance stays where it started.
        aia_image : array-like or None, optional
            A processed 2D AIA channel to show instead of the flux map, with
            the same HARP boxes and labels. It has to cover exactly the same
            crop the prediction used. The labels still show predicted flux, not
            AIA intensity. The image is shown with a linear 1–99 percentile
            stretch (or ``vmin``/``vmax``), pixel axes, and no colorbar.
            ``cmap="gray"`` usually looks best.
        harp_colors : dict or None, optional
            ``{harpnum: color}`` to set specific region colors, e.g. to match
            another plot. Any matplotlib color works: ``"tab:blue"``,
            ``"#56B4E9"``, or ``(0.2, 0.5, 0.8)``. Other regions keep their
            rank colors.

        Returns
        -------
        matplotlib.axes.Axes
            The map axes. Flux maps also get a colorbar.

        Raises
        ------
        ValueError
            If the map isn't a finite 2D array, has no positive values, or the
            color limits are invalid.
        """
        from matplotlib import pyplot as plt
        from matplotlib import patheffects
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Rectangle

        flux_map = np.asarray(self.flux_map)
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
        if harp_regions is None:
            harp_regions = self.harp_contributions
        regions = sorted(
            (region for region in _harp_region_fluxes(
                flux_map, harp_regions,
                self.flux_std_map,
            )
             if region["total_flux"] >= min_harp_flux),
            key=lambda region: (-region["total_flux"], region["harpnum"]),
        ) if show_harp_boxes and harp_regions else []
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
        if aia_image is None:
            title = []
            observed = self.ground_truth
            if observed is not None and np.isfinite(observed):
                title.append(f"Observed SXR-B: {observed:.2e} W/m²")
            if self.total_flux is not None and np.isfinite(self.total_flux):
                predicted = f"FOXES Prediction: {self.total_flux:.2e}"
                if self.flux_std_map is not None:
                    uncertainty = np.sqrt(np.sum(np.square(self.flux_std_map)))
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
                text = f'#{region["harpnum"]}\n{region["total_flux"]:.2e}'
                if show_harp_uncertainty and region["flux_std"] is not None:
                    text += f'\n± {region["flux_std"]:.2e}'
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

    def plot_uncertainty_map(self, *, ax=None, cmap="magma", vmin=None, vmax=None):
        """Plot ``flux_std_map`` contributions with a logarithmic color scale.

        Parameters
        ----------
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
            If the prediction has no uncertainty map (deterministic
            checkpoints), the map has no positive values, or the color limits
            are invalid. Nonpositive cells are masked.
        """
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm

        if self.flux_std_map is None:
            raise ValueError("This prediction has no patch uncertainty map (deterministic checkpoint)")
        flux_std_map = np.asarray(self.flux_std_map)
        if flux_std_map.ndim != 2 or not np.isfinite(flux_std_map).all():
            raise ValueError("flux_std_map must be a finite 2D array")
        positive = flux_std_map[flux_std_map > 0]
        if positive.size == 0:
            raise ValueError("flux_std_map must contain positive contributions")
        low, high = _log_limits(positive, vmin, vmax)
        if ax is None:
            _, ax = plt.subplots()
        artist = ax.imshow(
            np.ma.masked_less_equal(flux_std_map, 0),
            origin="lower", cmap=cmap,
            norm=LogNorm(vmin=low, vmax=high)
        )
        ax.set(xlabel="X (Patches)", ylabel="Y (Patches)")
        ax.figure.colorbar(artist, ax=ax, label="Patch flux standard-deviation contribution (W m$^{-2}$)")
        return ax


@dataclass(frozen=True)
class PredictionTimeSeries:
    """A set of predictions over time, for light curves.

    You'll normally get one from :meth:`FOXESModel.predict_time_series`. Times are
    the requested UTC times. Any time that was skipped during download just
    won't appear. When a HARP isn't present at some time, its light curve
    has a NaN there rather than a zero. Regional fluxes are computed as in
    :class:`Prediction`.

    Parameters
    ----------
    predictions : iterable of Prediction
        Predictions with metadata and no two sharing a timestamp. They're
        sorted by time and stored as a tuple.
    snapshot : Observation or None, optional
        A single AIA channel, shape ``(1, H, W)``, at one of the prediction
        times. It's only used for the image panel in :meth:`plot_light_curves`.
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

    def full_disk_light_curve(self):
        """Return the full-disk light curve as a DataFrame indexed by UTC time.

        Columns are ``total_flux`` (predicted) and ``ground_truth`` (measured
        GOES XRS-B, if available), both in W/m². Legacy series also get
        ``legacy_correction_enabled`` and ``flux_scale_factor`` columns.
        """
        rows = [{
            "timestamp": self._timestamp(p), "total_flux": p.total_flux,
            "ground_truth": p.ground_truth,
        } for p in self.predictions]
        frame = pd.DataFrame(rows, columns=["timestamp", "total_flux", "ground_truth"])
        if any(p.legacy_correction_enabled for p in self.predictions):
            frame["legacy_correction_enabled"] = [p.legacy_correction_enabled for p in self.predictions]
            frame["flux_scale_factor"] = [p.flux_scale_factor for p in self.predictions]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.set_index("timestamp")

    def harp_light_curves(self):
        """Return one light curve per active region, as ``{harpnum: DataFrame}``.

        Each DataFrame is indexed by UTC time and covers every prediction in
        the series, with ``total_flux`` and ``flux_std`` columns in
        W/m². If a region isn't present at a time, or there's no uncertainty,
        that value is NaN. Nothing is interpolated or filled with zeros.
        """
        #Get the index of the predictions (based on their timestamps)
        index = self.full_disk_light_curve().index
        # Collect all HARP numbers present across the series.
        ids = sorted({r["harpnum"] for p in self.predictions for r in p.harp_contributions or []})

        # Collect flux and uncertainty at every prediction time for each HARP.
        curves = {}
        for harpnum in ids:
            rows = []
            for prediction in self.predictions:
                region = next((r for r in prediction.harp_contributions or [] if r["harpnum"] == harpnum), {})
                rows.append([region.get("total_flux", np.nan), region.get("flux_std", np.nan)])
            curves[harpnum] = pd.DataFrame(
                rows, index=index, columns=["total_flux", "flux_std"], dtype=float,
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
            flux = _positive_flux(curves[harpnum]["total_flux"])
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

    def plot_light_curves(self, *, harpnums=None, ax=None, show_full_disk=False, show_observed=False,
             show_snapshot=False, snapshot_ax=None, aia_ax=None,
             show_uncertainty=True, min_peak_flux=0.0, legend="auto"):
        """Plot HARP light curves, optionally with the full-disk total and GOES.

        Every region in the series is plotted by default. The flux axis is
        logarithmic, so missing, zero, or negative values leave a gap in the
        line rather than being clipped to the bottom of the plot. Each HARP
        keeps the same color even when you filter regions out.

        **Uncertainty.** For Gaussian checkpoints, each region gets a shaded
        band of ± one ``flux_std``. A band whose lower edge would
        fall at or below zero is left out. Deterministic checkpoints get no
        bands. As in :class:`Prediction`, these are the model's uncertainty
        split across regions, not tested regional error bars.

        **Snapshot panels.** With ``show_snapshot=True``, two panels are added
        to the right of the curves: the flux map at the snapshot time (from
        :meth:`Prediction.plot_flux_map`, with HARP boxes in the same colors
        as the curves) and the plain AIA image. A dashed line marks the
        snapshot time on the curves. You need to have called
        ``predict_time_series(..., snapshot_time=...)`` first. The snapshot is
        the observation closest to that time, with the earlier one winning a
        tie.
        Only regions that are selected and present at that time get boxes.

        Parameters
        ----------
        harpnums : iterable of int or None, optional
            Which HARPs to plot, e.g. ``[1234, 5678]``. ``None`` plots all of
            them, and ``[]`` plots none (useful if you only want the totals).
            Wrap a single number in a list. Unknown numbers raise
            ``ValueError``.
        ax : matplotlib.axes.Axes or None, optional
            Axes for the light curves. If you're also showing the snapshot
            and want to supply your own axes, pass all three (``ax``,
            ``snapshot_ax``, ``aia_ax``), all different and in the same
            figure. Otherwise leave them out and the layout is made for you.
        show_full_disk, show_observed : bool, default False
            Add the predicted full-disk flux and the measured GOES XRS-B flux
            for comparison. ``min_peak_flux`` doesn't hide these.
        show_snapshot : bool, default False
            Add the flux map and AIA image panels described above.
        snapshot_ax, aia_ax : matplotlib.axes.Axes or None, optional
            Axes for the flux map and AIA image. See ``ax``.
        show_uncertainty : bool, default True
            Draw the uncertainty bands, and the ± lines on the snapshot's HARP
            labels, where the checkpoint provides them.
        min_peak_flux : float, default 0.0
            Hide HARPs whose highest flux over the whole series (not just at
            the snapshot) is below this value in W/m², e.g. ``1e-7``. Regions
            that never have a positive flux are always hidden. This applies to
            the snapshot labels as well, but it only affects the plot; the
            data in the series is unchanged.
        legend : {"auto", "full", "compact", "none"}, default "auto"
            Legend placed below the curves. ``"full"`` lists every region.
            ``"compact"`` shows a region count plus the comparison-line and
            marker keys, which is handy when the snapshot already labels the
            HARPs. ``"auto"`` picks compact when the snapshot is shown and full
            otherwise.

        Returns
        -------
        matplotlib.axes.Axes
            The light-curve axes. The snapshot panels, if any, are in the
            same figure.

        Examples
        --------
        >>> ax = series.plot_light_curves(show_snapshot=True,
        ...                  min_peak_flux=1e-7, legend="compact")
        """
        if legend not in {"auto", "full", "compact", "none"}:
            raise ValueError("legend must be 'auto', 'full', 'compact', or 'none'")
        curves = self.harp_light_curves()
        selected = self._select_harps(curves, harpnums, min_peak_flux)
        snapshot = self.snapshot if show_snapshot else None
        if show_snapshot and snapshot is None:
            raise ValueError("No AIA snapshot retained; use predict_time_series(..., snapshot_time=...)")
        #Creates subplots for the curves, flux map, and AIA image if show_snapshot is True. If ax is supplied, snapshot_ax and aia_ax must also be supplied and distinct. 
        ax, snapshot_ax, aia_ax = self._plot_axes(ax, snapshot_ax, aia_ax, show_snapshot)

        # Assign colors before filtering so each HARP keeps its identity.
        colors = {harpnum: _harp_rank_color(rank) for rank, harpnum in enumerate(curves)}
        self._plot_curves(ax, curves, selected, colors, show_full_disk, show_observed, show_uncertainty)
        if snapshot is not None:
            self._plot_snapshot_panels(ax, snapshot_ax, aia_ax, snapshot, selected, colors, show_uncertainty)
        self._plot_legend(ax, selected, legend, show_snapshot)
        return ax

    def _plot_curves(self, ax, curves, selected, colors, show_full_disk, show_observed, show_uncertainty):
        """Draw log-safe regional/comparison curves, uncertainty, and UTC axes."""
        has_uncertainty = False
        ax.set_yscale("log")
        for harpnum in selected:
            curve = curves[harpnum]
            flux = _positive_flux(curve["total_flux"])
            ax.plot(curve.index, flux, color=colors[harpnum], marker=".", markersize=2.5, label=f"HARP {harpnum}")
            if show_uncertainty:
                sigma = curve["flux_std"].to_numpy(dtype=float)
                lower, upper = flux - sigma, flux + sigma
                valid = np.isfinite(flux) & np.isfinite(sigma) & (sigma >= 0) & (lower > 0) & np.isfinite(upper)
                if valid.any():
                    ax.fill_between(
                        curve.index, np.where(valid, lower, np.nan), np.where(valid, upper, np.nan),
                        where=valid, color=colors[harpnum], alpha=0.18, linewidth=0,
                    )
                    has_uncertainty = True
        totals = self.full_disk_light_curve()
        if show_full_disk:
            ax.plot(totals.index, _positive_flux(totals["total_flux"]), color="black", marker=".", markersize=2.5, label="FOXES full disk")
        if show_observed:
            ax.plot(totals.index, _positive_flux(totals["ground_truth"]), color="0.5", linestyle="--", marker=".", markersize=2.5, label="Observed GOES XRS-B")
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
        regions = [r for r in prediction.harp_contributions or []
                   if r["harpnum"] in selected and np.isfinite(r["total_flux"]) and r["total_flux"] > 0]
        aia_ax.set_title(
            f"AIA {snapshot.metadata.wavelengths[0]} Å\n"
            f"{snapshot.metadata.timestamp:%Y-%m-%d %H:%M:%S} UTC"
        )
        ax.axvline(float(date2num(_utc_datetime(snapshot.metadata.timestamp))), color="0.5", linestyle=":", label="AIA snapshot")
        prediction.plot_flux_map(
            ax=snapshot_ax,
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


class AIAProcessor:
    """Download AIA images and turn them into a stack the model can use.

    Each stack has seven channels in the order 94, 131, 171, 193, 211, 304,
    and 335 Å. Processing uses the same ITI pipeline as training and gives
    you a channel-first ``float32`` array. Most of the time you'll want
    :class:`ObservationFetcher`, which wraps this class and also fetches
    HARP and GOES data.

    Parameters
    ----------
    email : str or None, optional
        Your registered JSOC email address. Defaults to the ``JSOC_EMAIL``
        environment variable.
    resolution : int, default 512
        Height and width of each processed channel, in pixels.
    tolerance : int, default 120
        How far, in seconds, each AIA frame can be from the requested time.
    n_workers : int, default 4
        Number of parallel download threads.
    """

    #: AIA channels in Å, in the order the model expects them.
    WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)
    # AIAStackDataset uses ITI's NormalizeRadiusEditor default crop padding.
    PADDING_FACTOR = 0.1

    def __init__(self, *, email=None, resolution=512, tolerance=120,
                 n_workers=4):
        """Set up JSOC downloading; see the class docstring."""
        if resolution <= 0:
            raise ValueError("resolution must be positive")

        self.email = _jsoc_email(email)
        self.resolution = int(resolution)
        self.tolerance = int(tolerance)
        self.n_workers = int(n_workers)

    def fetch(self, time):
        """Download and process all seven channels for one time.

        The raw FITS files go into a temporary folder that's deleted when
        the method returns, even if something fails along the way.

        Parameters
        ----------
        time : datetime.datetime, datetime.date, numpy.datetime64, or str
            Any format :func:`parse_timestamp` accepts.

        Returns
        -------
        numpy.ndarray
            ``float32`` array of shape ``(7, resolution, resolution)`` in
            :attr:`WAVELENGTHS` order.

        Raises
        ------
        TypeError
            If ``time`` isn't a type :func:`parse_timestamp` understands.
        FileNotFoundError
            If any channel has no frame within ``tolerance`` seconds.
        ValueError
            If processing returns the wrong shape or non-finite values.
        """
        from download.download_sdo import SDODownloader

        time = _utc_datetime(time).replace(tzinfo=None)

        with TemporaryDirectory(prefix="foxes-aia-") as temporary_path:
            #Download the seven AIA FITS files in parallel into a temporary directory
            downloader = SDODownloader(
                base_path=temporary_path, email=self.email,
                wavelengths=[str(w) for w in self.WAVELENGTHS],
                n_workers=self.n_workers, tolerance=self.tolerance,
            )
            downloader.downloadDate(time)
            stamp = time.isoformat("T", timespec="seconds")
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
        """Process AIA FITS files you already have on disk.

        This runs the same pipeline as :meth:`fetch`, just without the
        download, and your files are left untouched. Wrap the result with
        :meth:`Observation.from_array` to predict on it::

            paths = [Path("fits") / f"{w}.fits" for w in AIAProcessor.WAVELENGTHS]
            stack = processor.process_files(paths)
            observation = Observation.from_array(stack, timestamp="2023-08-01 12:00")

        Parameters
        ----------
        paths : iterable of str or pathlib.Path
            A list of exactly seven FITS files, one per channel, in
            :attr:`WAVELENGTHS` order. A folder name won't work here.

        Returns
        -------
        numpy.ndarray
            ``float32`` array of shape ``(7, resolution, resolution)``.
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


class HARPProcessor:
    """Look up HMI active regions (HARPs) and place their boxes on an AIA crop.

    Boxes come from the ``hmi.sharp_720s`` series on JSOC and are converted
    to percentages of the processed AIA image, so they line up with the
    flux map.
    """

    _LIMB_BBOX_EDGE_SAMPLES = 25

    def __init__(self, *, email=None, resolution=512, padding_factor=0.1,
                 bbox_expansion=0.0):
        """Configure JSOC access and the AIA crop geometry.

        Parameters
        ----------
        email : str or None, optional
            Your registered JSOC email address. Defaults to the
            ``JSOC_EMAIL`` environment variable.
        resolution : int, default 512
            Side length of the processed AIA image in pixels.
        padding_factor : float, default 0.1
            How far the AIA crop extends past the solar limb, as a fraction
            of the solar radius.
        bbox_expansion : float, default 0.0
            Fraction of each box's width and height added on every side
            (0.1 grows each side by 10%). Expanded boxes are clipped to the
            image.
        """
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if not np.isfinite(bbox_expansion) or bbox_expansion < 0:
            raise ValueError("bbox_expansion must be finite and nonnegative")
        self.email = _jsoc_email(email)
        self.resolution = int(resolution)
        self.padding_factor = float(padding_factor)
        self.bbox_expansion = float(bbox_expansion)
        self.SHARP_SERIES = "hmi.sharp_720s"
        self.SEARCH_WINDOW = timedelta(minutes=15)
        self._QUERY_KEYS = "HARPNUM,NOAA_ARS,T_REC,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX,RSUN_OBS,QUALITY"
        self._T_REC_FORMAT = "%Y.%m.%d_%H:%M:%S"

    def fetch(self, time: datetime | str) -> list[dict]:
        """Get the HARP regions closest in time to ``time``.

        Uses the nearest HARP record with good-quality regions. Each row
        keeps the original JSOC columns, including ``HARPNUM`` and the raw
        ``NOAA_ARS`` string. Regions visible in the crop also get
        ``top_pct``, ``left_pct``, ``width_pct``, and ``height_pct``.

        HARPs are a nice-to-have, so if the query fails or nothing usable
        turns up, you get an empty list instead of an error. An invalid
        timestamp still raises.

        Parameters
        ----------
        time : datetime.datetime, datetime.date, numpy.datetime64, or str
            Any format :func:`parse_timestamp` accepts. JSOC's own TAI
            record times aren't supported.

        Returns
        -------
        list of dict
        """
        timestamp = _utc_datetime(time).replace(tzinfo=None)
        try:
            df = self._query_sharp(timestamp, self.email)
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


class SXRProcessor:
    """Fetch measured GOES X-ray flux to compare against predictions.

    By default the fluxes from every GOES satellite with good data are
    averaged, the same way the training targets were built
    (see ``data.align_aia_sxr``). The average is a plain mean of the
    physical fluxes, taken before any log scaling.
    """

    def __init__(self, *, tolerance=120, satellite=None):
        """Set up the GOES lookup.

        Parameters
        ----------
        tolerance : float, default 120
            How far, in seconds, a GOES sample can be from the requested time.
        satellite : int or None, optional
            Use only this GOES satellite, given as a number (``18``, not
            ``"GOES-18"``). ``None`` averages all satellites with good data.
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

    def fetch(self, time) -> dict | None:
        """Get the measured XRS-A (0.5–4 Å) and XRS-B (1–8 Å) flux near a time.

        Finds the GOES sample closest to ``time`` that has good XRS-B data
        (quality flag zero and a positive flux), the same selection used for
        the training targets. If two samples are equally close, the earlier
        one wins. Only satellites reporting at that exact sample time are
        averaged. Values are used as the science product reports them, with
        no interpolation or extra scaling.

        Like HARPs, GOES data is optional, so a missing sample or failed
        query returns ``None`` instead of raising. An invalid timestamp still
        raises.

        Parameters
        ----------
        time : datetime.datetime, datetime.date, numpy.datetime64, or str
            Any format :func:`parse_timestamp` accepts.

        Returns
        -------
        dict or None
            ``xrsb_flux`` and ``xrsa_flux`` in W/m² (``xrsa_flux`` is
            ``None`` if no satellite had good XRS-A data), ``timestamp`` (the
            actual UTC sample time), ``time_offset_seconds`` (sample time minus
            requested time), ``satellites`` (which ones were averaged), and
            ``measurements`` (each satellite's own values).
        """
        timestamp = _utc_datetime(time)
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


def parse_timestamp(value):
    """Parse a date/time and return it as a timezone-aware UTC datetime.

    Every function in FOXES that takes a time runs it through this parser,
    so the rules here apply everywhere:

    - A date with no time means midnight.
    - A time with no timezone is treated as UTC.
    - For other timezones, give an explicit offset such as ``-04:00``.
      Abbreviations other than UTC and GMT (``EST``, for example) aren't
      recognized.

    Year-first formats work, and so do written-out month names. All-numeric
    dates like ``01/08/2023`` are rejected because the day/month order is
    ambiguous. Unix timestamps, partial dates ("August 2023"), relative
    dates, and NaT are rejected too.

    Parameters
    ----------
    value : datetime.datetime, datetime.date, numpy.datetime64, or str
        For example ``"2023-08-01"``, ``"2023-08-01T12:00:00Z"``,
        ``"2023/08/01 12:00"``, ``"2023.08.01_12:00:00_UTC"``,
        ``"August 1, 2023 12:00 UTC"``, or ``"2023-08-01T08:00:00-04:00"``.

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


def _resolve_checkpoint(checkpoint):
    """Return a checkpoint file path, downloading a released checkpoint by name."""
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path.resolve()
    relative = PRETRAINED_CHECKPOINTS.get(str(checkpoint))
    if relative is None:
        raise FileNotFoundError(
            f"{checkpoint} is not a file or a released checkpoint name "
            f"({', '.join(PRETRAINED_CHECKPOINTS)})"
        )
    cache = Path(os.environ.get("HELIO_FOXES_CACHE", "~/.cache/helio_foxes")).expanduser()
    target = cache / relative
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        _download(_CHECKPOINT_URL + relative, target)
    return target


def _download(url, target):
    """Download ``url`` to ``target`` with a progress bar, leaving nothing behind on failure."""
    from urllib.request import urlopen

    partial = target.with_name(target.name + ".part")
    try:
        with urlopen(url) as response, open(partial, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {target.name}") as bar:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    bar.update(len(chunk))
        with open(partial, "rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"{url} returned a Git LFS pointer instead of the checkpoint")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _jsoc_email(email):
    """Use the given JSOC email, falling back to the ``JSOC_EMAIL`` environment variable."""
    return email if email is not None else os.environ.get("JSOC_EMAIL")


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


def _make_aia_snapshot(image, observation, wavelength):
    """Copy one channel into a visualization-only Observation with its provenance."""
    metadata = observation.metadata
    return Observation(
        data=np.array(image, copy=True)[np.newaxis],
        metadata=ObservationMetadata(
            timestamp=metadata.timestamp, image_shape=(1, *image.shape),
            wavelengths=(wavelength,), padding_factor=metadata.padding_factor,
            harp_bbox_expansion=metadata.harp_bbox_expansion,
        ),
        harp_regions=observation.harp_regions, sxr=observation.sxr,
    )


def _positive_flux(values):
    """Return log-plottable values, leaving nonpositive/unknown samples as gaps."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


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


def _harp_region_fluxes(flux_map, regions, flux_std_map=None):
    """Copy HARP boxes and sum their flux and additive variance contributions."""
    flux_map = np.asarray(flux_map)
    patch_variance = None
    if flux_std_map is not None:
        uncertainty = np.asarray(flux_std_map, dtype=float)
        if uncertainty.ndim != 2 or uncertainty.shape != flux_map.shape:
            raise ValueError("flux_std_map must be 2D and match flux_map shape")
        if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
            raise ValueError("flux_std_map must be finite and nonnegative")
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
            "total_flux": _harp_box_patch_sum(
                flux_map, top=top, left=left, width=width, height=height,
            ),
            "flux_std": float(np.sqrt(_harp_box_patch_sum(
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

"""Download, predict from, and plot seven-channel AIA observations.

The public functions operate on channel-first arrays. Loading a checkpoint and
fetching data are separate so an application can reuse a model across requests.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

class DataProcessor:
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

        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace("Z", "+00:00"))
        if not isinstance(date, datetime):
            raise TypeError("date must be a datetime or ISO 8601 string")
        if date.tzinfo is not None:
            date = date.astimezone(timezone.utc).replace(tzinfo=None)

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


@dataclass(frozen=True)
class Prediction:
    """FOXES output for one AIA stack.

    Attributes
    ----------
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
    """

    flux: float
    flux_map: np.ndarray
    intervals: dict[str, tuple[float, float]] | None = None
    patch_uncertainty_map: np.ndarray | None = None


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
        with one. Do not apply that artifact in :class:`DataProcessor` too.

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

    def predict(self, image):
        """Predict global SXR flux and patch contributions from one AIA stack.

        Parameters
        ----------
        image : array-like
            Finite channel-first processed AIA stack in 94, 131, 171, 193,
            211, 304, 335 Å order. Shape is ``(7, H, W)`` with ``H`` and
            ``W`` determined by the checkpoint's patch size and grid.

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

        array = np.asarray(image)
        patch_size = int(self.model_kwargs["patch_size"])
        patch_count = int(self.model_kwargs["num_patches"])
        grid = int(np.sqrt(patch_count))
        channels = int(self.model_kwargs["num_channels"])
        expected = (channels, grid * patch_size, grid * patch_size)
        if array.shape != expected:
            raise ValueError(f"Expected processed AIA shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("AIA image contains non-finite values")
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
            flux=float(flux.item()), flux_map=flux_map,
            intervals=intervals, patch_uncertainty_map=patch_uncertainty,
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


def predict(image, *, model):
    """Run one processed AIA stack through a loaded FOXES model.

    Parameters
    ----------
    image : array-like
        Channel-first AIA stack; see :meth:`Predictor.predict` for its shape
        and channel order.
    model : Predictor
        Predictor returned by :func:`load_model`.

    Returns
    -------
    Prediction
        Full-disk flux, flux map, and optional uncertainty in W/m².
    """
    return model.predict(image)


def _log_limits(positive, vmin, vmax):
    """Resolve positive color limits, expanding a constant automatic range."""
    low = float(positive.min()) if vmin is None else float(vmin)
    high = float(positive.max()) if vmax is None else float(vmax)
    if vmin is None and vmax is None and low == high:
        low, high = low / 10, high * 10
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or low >= high:
        raise ValueError("vmin and vmax must be finite, positive, and vmin < vmax")
    return low, high


def plot_flux_map(prediction, *, ax=None, cmap="inferno", vmin=None, vmax=None):
    """Plot patch flux contributions with a logarithmic color scale.

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
    from matplotlib.colors import LogNorm

    flux_map = np.asarray(
        prediction.flux_map if isinstance(prediction, Prediction) else prediction
    )
    if flux_map.ndim != 2 or not np.isfinite(flux_map).all():
        raise ValueError("flux_map must be a finite 2D array")
    positive = flux_map[flux_map > 0]
    if positive.size == 0:
        raise ValueError("flux_map must contain positive contributions")
    low, high = _log_limits(positive, vmin, vmax)
    if ax is None:
        _, ax = plt.subplots()
    artist = ax.imshow(
        np.ma.masked_less_equal(flux_map, 0),
        origin="lower", cmap=cmap,
        norm=LogNorm(vmin=low, vmax=high)
    )
    ax.set(xlabel="Image x (pixels)", ylabel="Image y (pixels)")
    ax.figure.colorbar(artist, ax=ax, label="Flux contribution (W m$^{-2}$)")
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
    ax.set(xlabel="Image x (pixels)", ylabel="Image y (pixels)")
    ax.figure.colorbar(artist, ax=ax, label="Patch uncertainty contribution (W m$^{-2}$)")
    return ax

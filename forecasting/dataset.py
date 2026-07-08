from datetime import timedelta

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import glob


def _normalize_timestamp(ts: str) -> str:
    """Normalize timestamp strings with underscores instead of colons (cross-platform filenames)."""
    if 'T' in ts:
        date_part, time_part = ts.split('T', 1)
        return f"{date_part}T{time_part.replace('_', ':')}"
    return ts


class SXRLogNormTransform:
    """Picklable SXR log-normalization transform (replaces T.Lambda for spawn compatibility)."""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: float) -> float:
        return (np.log10(x + 1e-8) - self.mean) / self.std


class AIAGOESDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading paired AIA (EUV images) and GOES (SXR flux) data.

    This dataset prepares AIA multi-wavelength image patches and corresponding
    GOES soft X-ray (SXR) scalar flux values for regression or prediction tasks.

    Parameters
    ----------
    aia_dir : str or Path
        Directory containing AIA .npy files.
    sxr_dir : str, Path, or None
        Directory containing SXR .npy files. Only required when `only_prediction`
        is False (i.e. you have ground truth to compare against).
    wavelengths : list of int, optional
        AIA wavelengths to include (default: [94, 131, 171, 193, 211, 304]).
    sxr_transform : callable, optional
        Transform to normalize or preprocess SXR flux values.
    target_size : tuple of int, optional
        Target spatial dimensions for AIA images (default: (512, 512)).
    cadence : int, optional
        Time interval in minutes between samples (default: 1).
    reference_time : datetime or None
        Optional reference timestamp for temporal alignment.
    only_prediction : bool, optional
        If True, loads only AIA images without requiring SXR targets.
    """

    def __init__(self, aia_dir, sxr_dir, wavelengths=[94, 131, 171, 193, 211, 304, 335], sxr_transform=None,
                 target_size=(512, 512), cadence=1, reference_time=None, only_prediction=False):
        self.aia_dir = Path(aia_dir).resolve()
        self.sxr_dir = Path(sxr_dir).resolve() if sxr_dir else None
        if self.sxr_dir is None and not only_prediction:
            raise ValueError("sxr_dir is required unless only_prediction=True")
        self.wavelengths = wavelengths
        self.sxr_transform = sxr_transform
        self.target_size = target_size
        self.samples = []
        self.only_prediction = only_prediction
        self.cadence = timedelta(minutes=cadence)
        self.reference_time = reference_time

        # Check directories
        if not self.aia_dir.is_dir():
            raise FileNotFoundError(f"AIA directory not found: {self.aia_dir}")
        if self.sxr_dir is not None and not self.sxr_dir.is_dir():
            raise FileNotFoundError(f"SXR directory not found: {self.sxr_dir}")

        # Find matching files
        aia_files = sorted(glob.glob(str(self.aia_dir / "*.npy")))
        aia_files = [Path(f) for f in aia_files]

        # Collect valid samples
        valid_samples = []
        for f in aia_files:
            timestamp = f.stem
            timestamp_dt = pd.to_datetime(_normalize_timestamp(timestamp))

            if self.reference_time is None:
                self.reference_time = timestamp_dt
                aligned = True
            else:
                delta = (timestamp_dt - self.reference_time).total_seconds()
                aligned = (delta % self.cadence.total_seconds()) == 0

            if not aligned:
                continue

            if valid_samples and (
                    timestamp_dt - pd.to_datetime(_normalize_timestamp(valid_samples[-1]))).total_seconds() < self.cadence.total_seconds():
                continue

            if self.only_prediction:
                valid_samples.append(timestamp)
            elif (self.sxr_dir / f"{timestamp}.npy").exists():
                valid_samples.append(timestamp)

        self.samples = valid_samples

        if len(self.samples) == 0 and not self.only_prediction:
            raise ValueError("No valid sample pairs found")

    def __len__(self):
        """Return number of available samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single sample (AIA image and SXR value).

        Parameters
        ----------
        idx : int
            Index of sample.

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            (AIA image tensor [H, W, C], normalized SXR scalar tensor)
        """
        timestamp = self.samples[idx]
        aia_path = self.aia_dir / f"{timestamp}.npy"

        # Load AIA image as (7, H, W)
        try:
            all_wavelengths = [94, 131, 171, 193, 211, 304, 335]
            aia_img = np.load(aia_path)
            indices = [all_wavelengths.index(wav) for wav in self.wavelengths if wav in all_wavelengths]
            aia_img = aia_img[indices]
        except:
            print(f"Error loading AIA image from {aia_path}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self))

        # Convert to torch for transforms
        aia_img = torch.tensor(aia_img, dtype=torch.float32)  # (7, H, W)
        # Always output channel-last for model: (H, W, C)
        aia_img = aia_img.permute(1, 2, 0)  # (H, W, 7)

        # Load SXR value
        if not self.only_prediction:
            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            sxr_val = np.load(sxr_path)
        else:
            sxr_val = np.array([0])
        if sxr_val.size != 1:
            raise ValueError(f"SXR value has size {sxr_val.size}, expected scalar")
        sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
        if self.sxr_transform:
            sxr_val = self.sxr_transform(sxr_val)

        return aia_img, torch.tensor(sxr_val, dtype=torch.float32)

    def __gettimestamp__(self, idx):
        """
        Get the timestamp corresponding to a given index.

        Returns
        -------
        str
            Timestamp string of sample.
        """
        timestamp = self.samples[idx]
        return timestamp


class AIAGOESDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule wiring up train/val/test AIAGOESDataset splits.
    Used by train.py.

    Parameters
    ----------
    aia_train_dir, aia_val_dir, aia_test_dir : str
        Directories of AIA .npy files for each split.
    sxr_train_dir, sxr_val_dir, sxr_test_dir : str
        Directories of SXR .npy files for each split.
    sxr_norm : np.ndarray
        (mean, std) used to log-normalize SXR targets.
    batch_size, num_workers : int
    wavelengths : list of int
    """

    def __init__(self, aia_train_dir, aia_val_dir, aia_test_dir, sxr_train_dir, sxr_val_dir, sxr_test_dir,
                 sxr_norm, batch_size=64, num_workers=4, wavelengths=[94, 131, 171, 193, 211, 304, 335]):
        super().__init__()
        self.aia_train_dir = aia_train_dir
        self.aia_val_dir = aia_val_dir
        self.aia_test_dir = aia_test_dir
        self.sxr_train_dir = sxr_train_dir
        self.sxr_val_dir = sxr_val_dir
        self.sxr_test_dir = sxr_test_dir
        self.sxr_norm = sxr_norm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wavelengths = wavelengths

    def setup(self, stage=None):
        transform = SXRLogNormTransform(self.sxr_norm[0], self.sxr_norm[1])
        self.train_ds = AIAGOESDataset(aia_dir=self.aia_train_dir, sxr_dir=self.sxr_train_dir,
                                       sxr_transform=transform, wavelengths=self.wavelengths)
        self.val_ds = AIAGOESDataset(aia_dir=self.aia_val_dir, sxr_dir=self.sxr_val_dir,
                                     sxr_transform=transform, wavelengths=self.wavelengths)
        self.test_ds = AIAGOESDataset(aia_dir=self.aia_test_dir, sxr_dir=self.sxr_test_dir,
                                      sxr_transform=transform, wavelengths=self.wavelengths)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, prefetch_factor=4 if self.num_workers else None)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=4 if self.num_workers else None)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, prefetch_factor=1 if self.num_workers else None)



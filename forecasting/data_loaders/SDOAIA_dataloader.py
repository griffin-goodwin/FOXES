from datetime import timedelta

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
import glob
import os


class AIA_GOESDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading paired AIA (EUV images) and GOES (SXR flux) data.

    This dataset prepares AIA multi-wavelength image patches and corresponding
    GOES soft X-ray (SXR) scalar flux values for regression or prediction tasks.

    Parameters
    ----------
    aia_dir : str or Path
        Directory containing AIA .npy files.
    sxr_dir : str or Path
        Directory containing SXR .npy files.
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
    oversample : bool, optional
        If True, performs oversampling to balance flare/non-flare samples.
    flare_threshold : float, optional
        Threshold for labeling a sample as a flare (default: 1e-5).
    balance_strategy : {'upsample_minority', 'downsample_majority', 'balanced'}
        Strategy for balancing dataset classes.
    """

    def __init__(self, aia_dir, sxr_dir, wavelengths=[94, 131, 171, 193, 211, 304], sxr_transform=None,
                 target_size=(512, 512), cadence=1, reference_time=None, only_prediction=False, oversample=False,
                 flare_threshold=1e-5, balance_strategy='upsample_minority'):
        self.aia_dir = Path(aia_dir).resolve()
        self.sxr_dir = Path(sxr_dir).resolve()
        self.wavelengths = wavelengths
        self.sxr_transform = sxr_transform
        self.target_size = target_size
        self.samples = []
        self.only_prediction = only_prediction
        self.cadence = timedelta(minutes=cadence)
        self.reference_time = reference_time
        self.oversample = oversample
        self.flare_threshold = flare_threshold
        self.balance_strategy = balance_strategy  # 'upsample_minority', 'downsample_majority', 'balanced'

        # Check directories
        if not self.aia_dir.is_dir():
            raise FileNotFoundError(f"AIA directory not found: {self.aia_dir}")
        if not self.sxr_dir.is_dir():
            raise FileNotFoundError(f"SXR directory not found: {self.sxr_dir}")

        # Find matching files
        aia_files = sorted(glob.glob(str(self.aia_dir / "*.npy")))
        aia_files = [Path(f) for f in aia_files]

        # Collect valid samples
        valid_samples = []
        for f in aia_files:
            timestamp = f.stem
            timestamp_dt = pd.to_datetime(timestamp)

            if self.reference_time is None:
                self.reference_time = timestamp_dt
                aligned = True
            else:
                delta = (timestamp_dt - self.reference_time).total_seconds()
                aligned = (delta % self.cadence.total_seconds()) == 0

            if not aligned:
                continue

            if valid_samples and (
                    timestamp_dt - pd.to_datetime(valid_samples[-1])).total_seconds() < self.cadence.total_seconds():
                continue

            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            if sxr_path.exists() and not self.only_prediction:
                valid_samples.append(timestamp)
            elif self.only_prediction:
                valid_samples.append(timestamp)

        # Optional oversampling
        if self.oversample and not self.only_prediction:
            self.samples = self._apply_oversampling(valid_samples)
        else:
            self.samples = valid_samples

        if len(self.samples) == 0 and not self.only_prediction:
            raise ValueError("No valid sample pairs found")

    def _apply_oversampling(self, samples):
        """
        Balance the dataset by oversampling or downsampling based on SXR flare occurrence.

        Parameters
        ----------
        samples : list of str
            List of timestamp strings corresponding to available samples.

        Returns
        -------
        list of str
            Balanced list of timestamps.
        """
        import random

        flare_samples = []
        non_flare_samples = []

        for timestamp in samples:
            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            try:
                sxr_val = np.load(sxr_path)
                sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
                if sxr_val > self.flare_threshold:
                    flare_samples.append(timestamp)
                else:
                    non_flare_samples.append(timestamp)
            except:
                non_flare_samples.append(timestamp)

        print(f"Original distribution:")
        print(f"  Non-flare samples: {len(non_flare_samples)}")
        print(f"  Flare samples (>{self.flare_threshold}): {len(flare_samples)}")

        # Apply balancing strategy
        if self.balance_strategy == 'upsample_minority':
            if len(flare_samples) < len(non_flare_samples):
                target_count = len(non_flare_samples)
                balanced_samples = non_flare_samples.copy()
                balanced_samples.extend(self._oversample_to_count(flare_samples, target_count))
            else:
                target_count = len(flare_samples)
                balanced_samples = flare_samples.copy()
                balanced_samples.extend(self._oversample_to_count(non_flare_samples, target_count))

        elif self.balance_strategy == 'downsample_majority':
            if len(flare_samples) < len(non_flare_samples):
                target_count = len(flare_samples)
                balanced_samples = flare_samples.copy()
                balanced_samples.extend(random.sample(non_flare_samples, target_count))
            else:
                target_count = len(non_flare_samples)
                balanced_samples = non_flare_samples.copy()
                balanced_samples.extend(random.sample(flare_samples, target_count))

        elif self.balance_strategy == 'balanced':
            total_samples = len(flare_samples) + len(non_flare_samples)
            target_count = total_samples // 2
            balanced_samples = []
            balanced_samples.extend(self._oversample_to_count(flare_samples, target_count))
            balanced_samples.extend(self._oversample_to_count(non_flare_samples, target_count))
        else:
            raise ValueError(f"Unknown balance_strategy: {self.balance_strategy}")

        random.shuffle(balanced_samples)

        final_flare_count = sum(1 for ts in balanced_samples if self._is_flare_sample(ts))
        final_non_flare_count = len(balanced_samples) - final_flare_count

        print(f"Balanced distribution:")
        print(f"  Non-flare samples: {final_non_flare_count}")
        print(f"  Flare samples: {final_flare_count}")
        print(f"  Total samples: {len(balanced_samples)}")
        print(f"  Balance ratio: {final_flare_count / final_non_flare_count:.2f}")

        return balanced_samples

    def _oversample_to_count(self, samples, target_count):
        """
        Oversample a given list of samples to reach a target count.

        Parameters
        ----------
        samples : list
            List of sample identifiers (timestamps).
        target_count : int
            Target total number of samples after oversampling.

        Returns
        -------
        list
            Oversampled list of samples.
        """
        if len(samples) == 0:
            return []

        import random
        result = samples.copy()

        while len(result) < target_count:
            remaining_needed = target_count - len(result)
            to_add = min(remaining_needed, len(samples))
            result.extend(random.choices(samples, k=to_add))

        return result[:target_count]

    def _is_flare_sample(self, timestamp):
        """
        Determine if a timestamp corresponds to a flare sample.

        Returns
        -------
        bool
            True if flare intensity > flare_threshold, else False.
        """
        sxr_path = self.sxr_dir / f"{timestamp}.npy"
        try:
            sxr_val = np.load(sxr_path)
            sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
            return sxr_val > self.flare_threshold
        except:
            return False

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
        sxr_path = self.sxr_dir / f"{timestamp}.npy"

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


class AIA_GOESDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for managing AIA-GOES datasets.

    Handles creation of train, validation, and test DataLoaders
    with consistent preprocessing and sampling configurations.

    Parameters
    ----------
    aia_train_dir, aia_val_dir, aia_test_dir : str
        Directories containing AIA image data for train/val/test sets.
    sxr_train_dir, sxr_val_dir, sxr_test_dir : str
        Directories containing corresponding SXR data for train/val/test sets.
    sxr_norm : np.ndarray
        Normalization statistics for SXR flux (mean, std).
    batch_size : int, optional
        Batch size for dataloaders (default: 64).
    num_workers : int, optional
        Number of subprocesses for data loading.
    wavelengths : list[int], optional
        Wavelengths to include in the AIA images.
    cadence : int, optional
        Cadence interval in minutes.
    reference_time : datetime, optional
        Reference timestamp for alignment.
    only_prediction : bool, optional
        If True, disables SXR target loading.
    oversample : bool, optional
        Whether to apply class balancing during training.
    balance_strategy : str, optional
        Strategy for balancing dataset (default: 'upsample_minority').
    """

    def __init__(self, aia_train_dir, aia_val_dir, aia_test_dir, sxr_train_dir, sxr_val_dir, sxr_test_dir,
                 sxr_norm, batch_size=64, num_workers=4, wavelengths=[94, 131, 171, 193, 211, 304],
                 cadence=1, reference_time=None, only_prediction=False, oversample=False,
                 balance_strategy='upsample_minority'):
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
        self.cadence = cadence
        self.reference_time = reference_time
        self.only_prediction = only_prediction
        self.oversample = oversample
        self.balance_strategy = balance_strategy

    def setup(self, stage=None):
        """
        Initialize train, validation, and test datasets.
        """
        self.train_ds = AIA_GOESDataset(
            aia_dir=self.aia_train_dir,
            sxr_dir=self.sxr_train_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512),
            wavelengths=self.wavelengths,
            cadence=1,
            reference_time=None,
            only_prediction=False,
            oversample=self.oversample,
            flare_threshold=1e-5,
            balance_strategy=self.balance_strategy
        )

        self.val_ds = AIA_GOESDataset(
            aia_dir=self.aia_val_dir,
            sxr_dir=self.sxr_val_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512),
            wavelengths=self.wavelengths,
            cadence=1,
            reference_time=None,
            only_prediction=False,
            oversample=False,
            flare_threshold=1e-5,
            balance_strategy='upsample_minority'
        )

        self.test_ds = AIA_GOESDataset(
            aia_dir=self.aia_test_dir,
            sxr_dir=self.sxr_test_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512),
            wavelengths=self.wavelengths,
            cadence=1,
            reference_time=None,
            only_prediction=False,
            oversample=False,
            flare_threshold=1e-5,
            balance_strategy='upsample_minority'
        )

    def train_dataloader(self):
        """Return DataLoader for training set."""
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, prefetch_factor=4)

    def val_dataloader(self):
        """Return DataLoader for validation set."""
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, prefetch_factor=4)

    def test_dataloader(self):
        """Return DataLoader for test set."""
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, prefetch_factor=1)

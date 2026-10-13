"""Inference contracts for the maintained deterministic and Gaussian models."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from forecasting.inference import (
    evaluate_model_on_dataset, resolve_inference_device,
    select_inference_samples,
)
from forecasting.dataset import AIAGOESDataset
from forecasting.model import CoupledGaussianViTLocal, ViTLocal


class InferenceContractTest(unittest.TestCase):
    def test_inference_device_selection(self):
        with (patch('torch.cuda.is_available', return_value=False),
              patch('torch.backends.mps.is_available', return_value=True)):
            self.assertEqual(resolve_inference_device({}).type, 'mps')
            self.assertEqual(resolve_inference_device({'device': 'mps'}).type, 'mps')
            self.assertEqual(resolve_inference_device({'device': 'cpu'}).type, 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                resolve_inference_device({'device': 'cuda'})
        with (patch('torch.cuda.is_available', return_value=True),
              patch('torch.backends.mps.is_available', return_value=True)):
            self.assertEqual(resolve_inference_device({}).type, 'cuda')
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'MPS is unavailable'):
                resolve_inference_device({'device': 'mps'})
        with (patch('torch.cuda.is_available', return_value=False),
              patch('torch.backends.mps.is_available', return_value=False)):
            self.assertEqual(resolve_inference_device({}).type, 'cpu')
        with self.assertRaisesRegex(ValueError, 'device must be one of'):
            resolve_inference_device({'device': 'tpu'})

    def test_dataset_scan_reports_matching_progress(self):
        with TemporaryDirectory() as root:
            aia_dir = Path(root) / 'aia'
            sxr_dir = Path(root) / 'sxr'
            aia_dir.mkdir()
            sxr_dir.mkdir()
            stamps = ['2020-01-01T00_00_00', '2020-01-01T00_01_00']
            for stamp in stamps:
                np.save(aia_dir / f'{stamp}.npy', np.zeros(1))
            np.save(sxr_dir / f'{stamps[0]}.npy', np.array([1e-6]))
            output = StringIO()
            with redirect_stdout(output):
                dataset = AIAGOESDataset(
                    aia_dir, sxr_dir, show_scan_progress=True,
                )
            self.assertEqual(dataset.samples, stamps[:1])
            self.assertIn('Found 2 AIA files', output.getvalue())
            self.assertIn('Matched 1 eligible samples', output.getvalue())

    def test_seeded_sampling_keeps_timestamps_aligned(self):
        dataset = TensorDataset(torch.arange(10))
        dataset.samples = [f"timestamp_{index}" for index in range(10)]

        sampled, times = select_inference_samples(
            dataset, {'count': 4, 'seed': 7}
        )
        sampled_again, times_again = select_inference_samples(
            dataset, {'count': 4, 'seed': 7}
        )

        indices = [sampled[index][0].item() for index in range(len(sampled))]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(times, [dataset.samples[index] for index in indices])
        self.assertEqual(times, times_again)
        self.assertEqual(indices, [sampled_again[index][0].item()
                                   for index in range(len(sampled_again))])
        fractional, _ = select_inference_samples(
            dataset, {'fraction': 0.21, 'seed': 7}
        )
        self.assertEqual(len(fractional), 3)

    def test_sampling_rejects_invalid_settings(self):
        dataset = TensorDataset(torch.arange(3))
        dataset.samples = ['a', 'b', 'c']
        for settings in (
            {'count': 0}, {'count': 4}, {'fraction': 0},
            {'fraction': 1.1}, {'count': 1, 'fraction': 0.5},
            {'count': 1, 'seed': -1},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    select_inference_samples(dataset, settings)
        all_samples, times = select_inference_samples(dataset)
        self.assertIs(all_samples, dataset)
        self.assertEqual(times, dataset.samples)

    def test_maintained_model_output_shapes(self):
        architecture = dict(
            embed_dim=8, hidden_dim=16, num_channels=7,
            num_heads=2, num_layers=1, patch_size=1,
            num_patches=4, dropout=0.0, mask_mode='local',
            local_window=3,
        )
        normalization = np.array([-6.0, 1.0], dtype=np.float32)
        dataset = TensorDataset(
            torch.zeros(1, 2, 2, 7), torch.tensor([1e-6])
        )
        options = {'num_workers': 0, 'pin_memory': False, 'use_amp': False}

        for model_class, kwargs, expected_length in (
            (ViTLocal, {}, 5),
            (CoupledGaussianViTLocal, {'uncertainty_kwargs': {'enabled': True}}, 6),
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class(
                    model_kwargs={
                        **architecture,
                        **({'global_patch_adjustment': True}
                           if model_class is CoupledGaussianViTLocal else {}),
                    },
                    sxr_norm=normalization, **kwargs,
                )
                result = next(evaluate_model_on_dataset(
                    model, dataset, batch_size=1, times=['sample'],
                    config_data=options, save_weights=False,
                    input_size=2, patch_size=1, save_flux=False,
                ))
                self.assertEqual(len(result), expected_length)
                if model_class is CoupledGaussianViTLocal:
                    self.assertGreater(result[-1]['variance_normalized'], 0)


if __name__ == '__main__':
    unittest.main()

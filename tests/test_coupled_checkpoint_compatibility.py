"""Portable checkpoint checks for the consolidated core model classes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from forecasting.inference import _auto_checkpoint_model_class
from forecasting.model import (
    CoupledGaussianViTLocal,
    CoupledSingleQueryGaussianNLLViTLocal,
    ViTLocal,
)
from training.train import load_training_config


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_custom_global_head_config_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config = Path(temp_dir) / 'custom_global.yaml'
            custom_config.write_text(yaml.safe_dump({
                'base_config': str(Path(
                    'training/train_coupled_gaussian.yaml'
                ).resolve()),
                'vit_architecture': {
                    'global_attention_heads': 8,
                    'global_head_hidden_dim': 512,
                },
            }))
            config = load_training_config(custom_config)
        self.assertEqual(config['model_type'], 'coupled_uncertainty')
        self.assertNotIn('mean_parameterization', config)
        self.assertEqual(config['vit_architecture']['global_attention_heads'], 8)
        self.assertEqual(config['vit_architecture']['global_head_hidden_dim'], 512)

        architecture = dict(
            embed_dim=8, hidden_dim=16, num_channels=7,
            num_heads=2, num_layers=1, patch_size=1,
            num_patches=4, dropout=0.0, mask_mode='local',
            local_window=3, global_patch_adjustment=True,
            global_attention_heads=1, global_head_hidden_dim=12,
            global_attention_dropout=0.05,
        )
        model = CoupledGaussianViTLocal(
            model_kwargs=architecture,
            sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
            uncertainty_kwargs={'enabled': True},
        )
        self.assertEqual(model.model.global_attention.num_heads, 1)
        self.assertEqual(model.model.global_patch_head[1].out_features, 12)
        self.assertEqual(model.model.global_attention.dropout, 0.05)
        model.eval()
        with torch.no_grad():
            expected = model(torch.zeros(1, 2, 2, 7))[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'custom.ckpt'
            torch.save({
                'pytorch-lightning_version': pl.__version__,
                'hyper_parameters': {
                    **dict(model.hparams),
                    'contrast_prior_kwargs': {'channel_indices': []},
                },
                'state_dict': model.state_dict(),
            }, path)
            self.assertIs(
                _auto_checkpoint_model_class(torch.load(
                    path, map_location='cpu', weights_only=False,
                )),
                CoupledGaussianViTLocal,
            )
            restored = CoupledGaussianViTLocal.load_from_checkpoint(
                path, map_location='cpu', weights_only=False, strict=True,
            )
            restored.eval()
            with torch.no_grad():
                actual = restored(torch.zeros(1, 2, 2, 7))[0]
            torch.testing.assert_close(actual, expected)

    def test_checkpoint_architecture(self):
        self.assertIs(CoupledSingleQueryGaussianNLLViTLocal, CoupledGaussianViTLocal)

        with tempfile.TemporaryDirectory() as temp_dir:
            for model_class, layers in (
                (ViTLocal, 2),
                (CoupledGaussianViTLocal, 4),
                (CoupledGaussianViTLocal, 12),
            ):
                architecture = dict(
                    embed_dim=8, hidden_dim=16, num_channels=7,
                    num_heads=2, num_layers=layers, patch_size=1,
                    num_patches=4, dropout=0.0, mask_mode='local',
                    local_window=3,
                )
                if model_class is CoupledGaussianViTLocal:
                    architecture['global_patch_adjustment'] = True
                    kwargs = dict(uncertainty_kwargs={'enabled': True})
                else:
                    kwargs = {}
                model = model_class(
                    model_kwargs=architecture,
                    sxr_norm=np.array([-6.0, 1.0], dtype=np.float32),
                    **kwargs,
                )
                path = Path(temp_dir) / f'{model_class.__name__}-{layers}.ckpt'
                checkpoint = {
                    'pytorch-lightning_version': pl.__version__,
                    'hyper_parameters': dict(model.hparams),
                    'state_dict': model.state_dict(),
                }
                torch.save(checkpoint, path)
                self.assertIs(_auto_checkpoint_model_class(checkpoint), model_class)
                restored = model_class.load_from_checkpoint(
                    path, map_location='cpu', weights_only=False, strict=True,
                )
                self.assertEqual(len(restored.model.transformer_blocks), layers)
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))


if __name__ == '__main__':
    unittest.main()

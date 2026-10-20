import numpy as np
import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

from forecasting.dataset import AIAGOESDataModule
from forecasting.model import CoupledPinballViTLocal, normalize_sxr
from forecasting.sampling import (
    FLARE_CLASS_NAMES, StratifiedFlareClassSampler, training_flare_class_indices,
)
from training.train import load_training_config


def class_groups(counts):
    edges = np.cumsum([0, *counts])
    groups = {name: list(range(edges[i], edges[i + 1]))
              for i, name in enumerate(FLARE_CLASS_NAMES)}
    return groups, np.repeat(np.arange(4), counts)


def batches(sampler):
    return np.asarray(list(sampler)).reshape(-1, sampler.batch_size)


def test_balanced_batches_reuse_exhausted_classes():
    groups, labels = class_groups([40, 8, 3, 1])
    sampler = StratifiedFlareClassSampler(range(len(labels)), groups, batch_size=8, seed=42)
    selected = batches(sampler)
    assert selected.shape == (7, 8)
    assert len(sampler) == 56
    for batch in selected:
        np.testing.assert_array_equal(np.bincount(labels[batch], minlength=4), [2, 2, 2, 2])
    # The only X example is deliberately reused, including within a batch.
    assert np.count_nonzero(selected == groups['x_class'][0]) == 14
    assert len(set(selected[labels[selected] == 0])) == 14


def test_batch_remainders_rotate_and_epoch_seed_is_reproducible():
    groups, labels = class_groups([20, 20, 20, 20])
    sampler = StratifiedFlareClassSampler(
        range(80), groups, batch_size=7, seed=3, batches_per_epoch=4,
    )
    first = batches(sampler)
    np.testing.assert_array_equal(first, batches(sampler))
    for batch in first:
        assert sorted(np.bincount(labels[batch], minlength=4)) == [1, 2, 2, 2]
    np.testing.assert_array_equal(np.bincount(labels[first].ravel(), minlength=4), [7] * 4)
    sampler.set_epoch(1)
    assert not np.array_equal(first, batches(sampler))
    duplicate = StratifiedFlareClassSampler(range(80), groups, 7, seed=3, batches_per_epoch=4)
    duplicate.set_epoch(1)
    np.testing.assert_array_equal(batches(sampler), batches(duplicate))


def test_distributed_ranks_get_balanced_distinct_draws_before_reuse():
    groups, labels = class_groups([20] * 4)
    ranks = [StratifiedFlareClassSampler(
        range(80), groups, 8, seed=42, batches_per_epoch=2, num_replicas=2, rank=rank,
    ) for rank in range(2)]
    selected = [batches(sampler) for sampler in ranks]
    assert not set(selected[0].ravel()) & set(selected[1].ravel())
    for rank_batches in selected:
        for batch in rank_batches:
            np.testing.assert_array_equal(np.bincount(labels[batch], minlength=4), [2] * 4)
    automatic = StratifiedFlareClassSampler(range(80), groups, 8, num_replicas=2, rank=0)
    assert automatic.batches_per_epoch == 5


def test_sampler_detects_the_initialized_distributed_rank(monkeypatch):
    groups, _ = class_groups([20] * 4)
    monkeypatch.setattr('forecasting.sampling.dist.is_available', lambda: True)
    monkeypatch.setattr('forecasting.sampling.dist.is_initialized', lambda: True)
    monkeypatch.setattr('forecasting.sampling.dist.get_world_size', lambda: 2)
    monkeypatch.setattr('forecasting.sampling.dist.get_rank', lambda: 1)
    automatic = StratifiedFlareClassSampler(range(80), groups, 8, seed=42)
    explicit = StratifiedFlareClassSampler(range(80), groups, 8, seed=42,
                                            num_replicas=2, rank=1)
    assert automatic.rank == 1
    assert automatic.num_replicas == 2
    np.testing.assert_array_equal(batches(automatic), batches(explicit))


@pytest.mark.parametrize('options, message', [
    ({'batch_size': 3}, 'batch_size'),
    ({'seed': -1}, 'seed'),
    ({'batches_per_epoch': 0}, 'batches_per_epoch'),
    ({'batches_per_epoch': True}, 'batches_per_epoch'),
])
def test_invalid_sampler_options_fail_clearly(options, message):
    groups, _ = class_groups([2] * 4)
    with pytest.raises(ValueError, match=message):
        StratifiedFlareClassSampler(range(8), groups, **{'batch_size': 8, **options})


def test_missing_class_fails_instead_of_silently_omitting_it():
    groups, _ = class_groups([10, 3, 1, 0])
    with pytest.raises(ValueError, match='empty bins.*x_class'):
        StratifiedFlareClassSampler(range(14), groups, 8)


def test_target_scan_uses_raw_flux_boundaries_without_loading_images(tmp_path):
    class MetadataOnlyDataset:
        sxr_dir = tmp_path
        samples = [str(i) for i in range(5)]

        def __getitem__(self, index):
            raise AssertionError('Grouping must not load AIA images')

    for i, flux in enumerate([1e-8, 1e-7, 1e-6, 1e-5, 1e-4]):
        np.save(tmp_path / f'{i}.npy', flux)
    assert training_flare_class_indices(MetadataOnlyDataset(), show_progress=False) == {
        'quiet': [0, 1], 'c_class': [2], 'm_class': [3], 'x_class': [4],
    }
    np.save(tmp_path / '0.npy', np.nan)
    with pytest.raises(ValueError, match='finite and positive'):
        training_flare_class_indices(MetadataOnlyDataset(), show_progress=False)


class MemoryDataModule(AIAGOESDataModule):
    def setup(self, stage=None):
        pass  # Tests inject small datasets without 512x512 AIA files.


def small_data_module(tmp_path, enabled=True):
    _, labels = class_groups([9, 4, 2, 1])
    raw = torch.tensor([1e-7, 2e-6, 2e-5, 2e-4])[labels]
    norm = np.array([-6., 1.], dtype=np.float32)
    dataset = TensorDataset(torch.rand(len(raw), 2, 2, 1), normalize_sxr(raw, torch.tensor(norm)))
    dataset.samples = [str(i) for i in range(len(raw))]
    dataset.sxr_dir = tmp_path
    for i, value in enumerate(raw):
        np.save(tmp_path / f'{i}.npy', float(value))
    data = MemoryDataModule('', '', '', '', '', '', norm, batch_size=8, num_workers=0,
                            stratified_batches=enabled, sampling_seed=42)
    data.train_ds = data.val_ds = data.test_ds = dataset
    return data


def test_data_module_toggle_caches_metadata_and_keeps_validation_natural(tmp_path, monkeypatch):
    data = small_data_module(tmp_path)
    assert isinstance(data.train_dataloader().sampler, StratifiedFlareClassSampler)
    monkeypatch.setattr('forecasting.dataset.training_flare_class_indices',
                        lambda *args, **kwargs: pytest.fail('Class indices should be cached'))
    assert isinstance(data.train_dataloader().sampler, StratifiedFlareClassSampler)
    assert isinstance(data.val_dataloader().sampler, SequentialSampler)
    assert isinstance(data.test_dataloader().sampler, SequentialSampler)
    data.stratified_batches = False
    assert isinstance(data.train_dataloader().sampler, RandomSampler)
    config = load_training_config('training/train_coupled_pinball.yaml')
    assert config['sampling']['stratified_batches'] is True


def test_lightning_preserves_distributed_sampler(tmp_path):
    data = small_data_module(tmp_path)
    loader = data.train_dataloader()
    trainer = Trainer(accelerator='cpu', devices=2, strategy='ddp_spawn',
                      logger=False, enable_checkpointing=False, enable_progress_bar=False)
    assert not trainer._data_connector._requires_distributed_sampler(loader)


def test_lightning_training_advances_stratified_epoch_and_logs_median_metrics(tmp_path):
    data = small_data_module(tmp_path)
    epochs = []

    class CheckBatches(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            sampler = trainer.train_dataloader.sampler
            assert isinstance(sampler, StratifiedFlareClassSampler)
            epochs.append((sampler.epoch, batches(sampler)))

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            target = batch[1]
            # All four class levels occur, twice each, in every training batch.
            _, counts = torch.unique(target, return_counts=True)
            torch.testing.assert_close(counts, torch.tensor([2, 2, 2, 2]))

    model = CoupledPinballViTLocal(
        dict(embed_dim=8, hidden_dim=16, num_channels=1, num_heads=1, num_layers=1,
             patch_size=1, num_patches=4, dropout=0., mask_mode='local',
             local_window=3, global_patch_adjustment=True), data.sxr_norm,
    )
    trainer = Trainer(accelerator='cpu', devices=1, max_epochs=2,
                      logger=CSVLogger(tmp_path), callbacks=[CheckBatches()],
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, log_every_n_steps=1)
    trainer.fit(model, datamodule=data)
    assert [epoch for epoch, _ in epochs] == [0, 1]
    assert not np.array_equal(epochs[0][1], epochs[1][1])
    assert torch.isfinite(trainer.callback_metrics['val/mse'])
    assert torch.isfinite(trainer.callback_metrics['val/rmse'])

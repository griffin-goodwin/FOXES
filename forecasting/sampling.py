"""Training samplers for paired AIA/SXR datasets."""

import math

import numpy as np
import torch.distributed as dist
from torch.utils.data import DistributedSampler


FLARE_CLASS_NAMES = ('quiet', 'c_class', 'm_class', 'x_class')
FLARE_CLASS_THRESHOLDS = (1e-6, 1e-5, 1e-4)


def training_flare_class_indices(dataset, show_progress=True):
    """Group training indices using raw scalar SXR files, without loading AIA."""
    groups = {name: [] for name in FLARE_CLASS_NAMES}
    if show_progress:
        print('Grouping training targets for stratified quiet/C/M/X batches...', flush=True)
    for index, timestamp in enumerate(dataset.samples):
        path = dataset.sxr_dir / f'{timestamp}.npy'
        value = np.load(path, allow_pickle=False)
        if value.size != 1:
            raise ValueError(f'Expected one SXR value in {path}, found {value.size}')
        flux = float(value.reshape(-1)[0])
        if not np.isfinite(flux) or flux <= 0:
            raise ValueError(f'SXR flux must be finite and positive in {path}: {flux!r}')
        class_index = int(np.searchsorted(FLARE_CLASS_THRESHOLDS, flux, side='right'))
        groups[FLARE_CLASS_NAMES[class_index]].append(index)
        if show_progress and (index + 1) % 10_000 == 0:
            print(f'Grouped {index + 1:,}/{len(dataset):,} training targets...', flush=True)
    if show_progress:
        counts = {name: len(indices) for name, indices in groups.items()}
        print(f'Stratified training class counts: {counts}', flush=True)
    return groups


class StratifiedFlareClassSampler(DistributedSampler):
    """Emit full batches with nearly equal quiet/C/M/X counts on every rank.

    Consecutive ``batch_size`` indices form one batch. Each class is shuffled
    and consumed in cycles, reshuffling whenever its pool runs out. Indices
    are assigned across ranks from one shared sequence before local shuffling.
    Reuse across ranks can occur when a class pool is exhausted.

    The default epoch is ``ceil(N / (batch_size * world_size))`` batches per
    rank. This keeps the number of draws close to an ordinary epoch; it does
    not promise to visit every majority-class example in each epoch.

    Inheriting DistributedSampler prevents Lightning from replacing this
    sampler and lets Lightning call ``set_epoch`` for reproducible reshuffling.
    """

    def __init__(self, dataset, class_indices, batch_size, seed=0,
                 batches_per_epoch=None, num_replicas=None, rank=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 4:
            raise ValueError('Stratified batching requires batch_size >= 4 per rank')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('Stratified sampling seed must be a nonnegative integer')
        if batches_per_epoch is not None and (
            isinstance(batches_per_epoch, bool) or not isinstance(batches_per_epoch, int)
            or batches_per_epoch <= 0
        ):
            raise ValueError('batches_per_epoch must be a positive integer or null')
        if set(class_indices) != set(FLARE_CLASS_NAMES):
            raise ValueError('Stratified batching requires quiet/C/M/X index groups')
        empty = [name for name in FLARE_CLASS_NAMES if len(class_indices[name]) == 0]
        if empty:
            raise ValueError(f'Stratified batching requires every training class; empty bins: {empty}')
        self.class_indices = tuple(
            np.asarray(class_indices[name], dtype=np.int64) for name in FLARE_CLASS_NAMES
        )
        if any((indices < 0).any() or (indices >= len(dataset)).any()
               for indices in self.class_indices):
            raise ValueError('Class indices must refer to entries in the training dataset')
        distributed = dist.is_available() and dist.is_initialized()
        if num_replicas is None:
            num_replicas = dist.get_world_size() if distributed else 1
        if rank is None:
            rank = dist.get_rank() if distributed else 0
        if isinstance(num_replicas, bool) or not isinstance(num_replicas, int) or num_replicas < 1:
            raise ValueError('num_replicas must be a positive integer')
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch or math.ceil(
            len(dataset) / (batch_size * num_replicas)
        )
        self.num_samples = self.batches_per_epoch * batch_size
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        local_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, self.rank, 1]),
        )
        orders = [rng.permutation(indices) for indices in self.class_indices]
        offsets = [0] * len(FLARE_CLASS_NAMES)

        def draw(class_index, count):
            chunks = []
            while count:
                if offsets[class_index] == len(orders[class_index]):
                    orders[class_index] = rng.permutation(self.class_indices[class_index])
                    offsets[class_index] = 0
                start = offsets[class_index]
                take = min(count, len(orders[class_index]) - start)
                chunks.append(orders[class_index][start:start + take])
                offsets[class_index] += take
                count -= take
            return np.concatenate(chunks)

        base, remainder = divmod(self.batch_size, len(FLARE_CLASS_NAMES))
        for step in range(self.batches_per_epoch):
            quotas = np.full((self.num_replicas, len(FLARE_CLASS_NAMES)), base)
            for rank in range(self.num_replicas):
                for extra in range(remainder):
                    quotas[rank, (step + rank + extra) % len(FLARE_CLASS_NAMES)] += 1
            batch = []
            for class_index in range(len(FLARE_CLASS_NAMES)):
                # Every rank makes the same global draws, then takes its slice.
                selected = draw(class_index, int(quotas[:, class_index].sum()))
                start = int(quotas[:self.rank, class_index].sum())
                count = int(quotas[self.rank, class_index])
                batch.extend(selected[start:start + count].tolist())
            local_rng.shuffle(batch)
            yield from batch

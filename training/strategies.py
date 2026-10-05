"""Distributed training strategies used by the training entry point."""

from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel


class TrainingStreamDDPStrategy(DDPStrategy):
    """Construct DDP on the current stream, also used by our training loop.

    Lightning 2.6 creates DDP on a temporary CUDA stream. DDP retains
    AccumulateGrad nodes created there, leading to stream mismatch warnings
    during backward. Our eager training does not need that side stream.
    """

    def _setup_model(self, model):
        return DistributedDataParallel(
            module=model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

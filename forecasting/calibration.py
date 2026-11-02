"""Opt-in physical-flux correction for checkpoints trained on legacy SXR targets."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re


def _utc_timestamp(value):
    """Read an ISO timestamp, including the timestamp stems used by NPY files."""
    if value is None:
        raise ValueError("A timestamp is required when legacy_model=True")
    if not isinstance(value, datetime):
        text = str(value).removesuffix(".npy")
        text = re.sub(r"([T ])(\d{2})_(\d{2})_(\d{2})", r"\1\2:\3:\4", text)
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Legacy correction requires an ISO timestamp, got {value!r}") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class LegacyCalibration:
    """Apply 0.7 strictly before a UTC cutoff; otherwise leave flux unchanged.

    This is a user-selected approximation for old checkpoints, including the
    mixed-satellite era. It does not modify weights or training normalization.
    Legacy checkpoints are deterministic. Only total and patch flux are scaled;
    attention weights and measured GOES values are not scaled.
    """

    enabled: bool = False
    cutoff: datetime | str = "2016-11-19"

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("legacy_model must be a boolean")
        object.__setattr__(self, "cutoff", _utc_timestamp(self.cutoff))

    def factor(self, timestamp):
        if not self.enabled:
            return 1.0
        return 0.7 if _utc_timestamp(timestamp) < self.cutoff else 1.0

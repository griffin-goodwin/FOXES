"""Small Python API for FOXES inference and visualization.

Fetch an :class:`Observation`, predict on it with a :class:`FOXESModel`, and
read or plot the resulting :class:`Prediction`. See :mod:`helio_foxes.api`.
"""

from .api import (
    # 1. Get data
    ObservationFetcher, Observation, load_observations,
    # 2. Load a model and predict
    FOXESModel,
    # 3. Results
    Prediction, PredictionTimeSeries,
    # Lower-level building blocks
    AIAProcessor, HARPProcessor, SXRProcessor, parse_timestamp,
)

__all__ = [
    "ObservationFetcher", "Observation", "load_observations",
    "FOXESModel",
    "Prediction", "PredictionTimeSeries",
    "AIAProcessor", "HARPProcessor", "SXRProcessor", "parse_timestamp",
]

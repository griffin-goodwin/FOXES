# API reference

Use [Getting started](python_api.md) for examples. The reference below reads
signatures, defaults, and docstrings directly from the code. Each definition
links to its source.

```{eval-rst}
.. currentmodule:: helio_foxes

.. autosummary::
   :nosignatures:

   Observation
   ObservationMetadata
   ObservationProcessor
   AIAProcessor
   HARPProcessor
   SXRProcessor
   Prediction
   Predictor
   load_model
   predict
   plot_flux_map
   plot_uncertainty_map
   parse_timestamp
   PredictionSeries
   load_observations
```

## Observation

```{eval-rst}
.. autoclass:: helio_foxes.Observation
   :members:
   :member-order: bysource
```

## ObservationMetadata

```{eval-rst}
.. autoclass:: helio_foxes.ObservationMetadata
   :members:
   :member-order: bysource
```

## ObservationProcessor

```{eval-rst}
.. autoclass:: helio_foxes.ObservationProcessor
   :members:
   :member-order: bysource
   :class-doc-from: both
```

## AIAProcessor

```{eval-rst}
.. autoclass:: helio_foxes.AIAProcessor
   :members:
   :member-order: bysource
```

## HARPProcessor

```{eval-rst}
.. autoclass:: helio_foxes.HARPProcessor
   :members:
   :member-order: bysource
   :class-doc-from: both
```

## SXRProcessor

```{eval-rst}
.. autoclass:: helio_foxes.SXRProcessor
   :members:
   :member-order: bysource
   :class-doc-from: both
```

## Prediction

```{eval-rst}
.. autoclass:: helio_foxes.Prediction
   :members:
   :member-order: bysource
```

## Predictor

```{eval-rst}
.. autoclass:: helio_foxes.Predictor
   :members:
   :member-order: bysource
```

## load_model

```{eval-rst}
.. autofunction:: helio_foxes.load_model
```

## predict

```{eval-rst}
.. autofunction:: helio_foxes.predict
```

## plot_flux_map

```{eval-rst}
.. autofunction:: helio_foxes.plot_flux_map
```

## plot_uncertainty_map

```{eval-rst}
.. autofunction:: helio_foxes.plot_uncertainty_map
```

## parse_timestamp

```{eval-rst}
.. autofunction:: helio_foxes.parse_timestamp
```

## PredictionSeries

```{eval-rst}
.. autoclass:: helio_foxes.PredictionSeries
   :members:
   :member-order: bysource
```

## load_observations

```{eval-rst}
.. autofunction:: helio_foxes.load_observations
```

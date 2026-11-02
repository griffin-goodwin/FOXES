# API Reference

Use [Getting Started](python_api.md) for examples. The reference below reads
signatures, defaults, and docstrings directly from the code. Each definition
links to its source.

The API is a short pipeline, and this page follows it in order:

| Step | Use | You get |
| --- | --- | --- |
| 1. Get data | {py:class}`~helio_foxes.ObservationFetcher`, {py:func}`~helio_foxes.load_observations`, or {py:meth}`Observation.from_array <helio_foxes.Observation.from_array>` | {py:class}`~helio_foxes.Observation` |
| 2. Load a model | {py:class}`FOXESModel(checkpoint) <helio_foxes.FOXESModel>` | {py:class}`~helio_foxes.FOXESModel` |
| 3. Predict | {py:meth}`FOXESModel.predict <helio_foxes.FOXESModel.predict>` / {py:meth}`~helio_foxes.FOXESModel.predict_time_series` | {py:class}`~helio_foxes.Prediction` / {py:class}`~helio_foxes.PredictionTimeSeries` |
| 4. Look at the results | attributes and `plot_*` methods on the result | numbers and figures |

The AIA, HARP, and SXR processors at the end are the building blocks the
fetcher uses; most people won't need them directly.

```{eval-rst}
.. currentmodule:: helio_foxes

.. autosummary::
   :nosignatures:

   ObservationFetcher
   Observation
   load_observations
   FOXESModel
   Prediction
   PredictionTimeSeries
   AIAProcessor
   HARPProcessor
   SXRProcessor
   parse_timestamp
```

## 1. Get Data

### ObservationFetcher

```{eval-rst}
.. autoclass:: helio_foxes.ObservationFetcher
   :members:
   :member-order: bysource
   :class-doc-from: both
```

### Observation

```{eval-rst}
.. autoclass:: helio_foxes.Observation
   :members:
   :member-order: bysource
```

#### ObservationMetadata

```{eval-rst}
.. autoclass:: helio_foxes.api.ObservationMetadata
   :members:
   :member-order: bysource
```

### load_observations

```{eval-rst}
.. autofunction:: helio_foxes.load_observations
```

## 2. Load a Model and Predict

### FOXESModel

```{eval-rst}
.. autoclass:: helio_foxes.FOXESModel
   :members:
   :member-order: bysource
```

## 3. Results and Plotting

### Prediction

```{eval-rst}
.. autoclass:: helio_foxes.Prediction
   :members:
   :member-order: bysource
   :exclude-members: observed_flux
```

### PredictionTimeSeries

```{eval-rst}
.. autoclass:: helio_foxes.PredictionTimeSeries
   :members:
   :member-order: bysource
```

## Lower-Level Building Blocks

### AIAProcessor

```{eval-rst}
.. autoclass:: helio_foxes.AIAProcessor
   :members:
   :member-order: bysource
```

### HARPProcessor

```{eval-rst}
.. autoclass:: helio_foxes.HARPProcessor
   :members:
   :member-order: bysource
   :class-doc-from: both
```

### SXRProcessor

```{eval-rst}
.. autoclass:: helio_foxes.SXRProcessor
   :members:
   :member-order: bysource
   :class-doc-from: both
```

### parse_timestamp

```{eval-rst}
.. autofunction:: helio_foxes.parse_timestamp
```

# Results

These numbers come from the updated `model_assets/UncertaintyModel.ckpt`,
with the `context` uncertainty head and the exact variance conversion described
in [Uncertainties](uncertainties.md). The saved evaluation covers the held-out
test set: **100,058 timestamps** between **2023-08-01 00:00** and
**2025-09-26 20:21 UTC**. The summary CSVs and figures below are from
`model_assets/results/evaluation/`.

## Metrics

| Group | Samples | RMSE (dex) | MAE (dex) | R² | Pearson correlation | 68% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 100,058 | 0.0895 | 0.0597 | 0.9712 | 0.9856 | 66.74% | 91.42% |
| B | 13,589 | 0.0486 | 0.0410 | 0.5936 | 0.8915 | 66.82% | 96.34% |
| C | 63,001 | 0.0745 | 0.0509 | 0.9376 | 0.9697 | 65.07% | 89.65% |
| M | 22,386 | 0.1282 | 0.0906 | 0.7320 | 0.8784 | 71.85% | 93.64% |
| X | 1,082 | 0.2197 | 0.1657 | −0.1718 | 0.6296 | 57.30% | 87.34% |

Errors are computed on `log10(flux)`, so they're in dex. Coverage is the
fraction of measured values that land within 1 (68%) or 1.96 (95%) predicted
standard deviations, measured in the checkpoint's normalized
`log10(flux + 1e-8)` space.

Full summaries:

- {download}`Regression metrics <../model_assets/results/evaluation/metrics/performance_metrics.csv>`
- {download}`Uncertainty metrics <../model_assets/results/evaluation/metrics/uncertainty_metrics.csv>`

## Predicted Versus Measured Flux

```{figure} ../model_assets/results/evaluation/plots/regression_plot.png
:alt: Density of predicted versus measured SXR flux, with a perfect-agreement line and class-specific MAE band.
:width: 100%

Full-disk flux in W/m². The red line is perfect agreement, and the shading
shows the mean absolute error (in log space) for each flare class.
```

## Calibration

```{figure} ../model_assets/results/evaluation/plots/uncertainty_calibration.png
:alt: Nominal versus empirical central interval coverage, overall and by flare class.
:width: 100%

How often the measured flux actually falls inside each nominal central
interval, overall and by flare class.
```

## Prediction Intervals

```{figure} ../model_assets/results/evaluation/plots/prediction_intervals.png
:alt: Predicted versus measured flux with asymmetric nominal 68% and 95% intervals for a class-balanced sample.
:width: 100%

The same 48 examples (up to 12 per class) are shown with nominal 68% intervals
on the left and 95% intervals on the right. The table's coverage values use
all 100,058 timestamps.
```

To run the same evaluation on your own predictions, see the
[Evaluate Predictions](workflows.md#evaluate-predictions) workflow.

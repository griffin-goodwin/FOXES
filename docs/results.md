# Results

These saved results are for `model_assets/UncertaintyModel.ckpt`, evaluated on
**100,058 timestamps** from **2023-08-01 00:00:00** through
**2025-09-26 20:21:00 UTC**. They measure full-disk predictions at individual
timestamps, rather than independent flare events.

## Metrics

| Group | Samples | RMSE (dex) | MAE (dex) | R² | Pearson correlation | 68% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 100,058 | 0.0933 | 0.0636 | 0.9687 | 0.9843 | 64.46% | 90.04% |
| B | 13,589 | 0.0452 | 0.0379 | 0.6492 | 0.8934 | 59.58% | 95.59% |
| C | 63,001 | 0.0801 | 0.0568 | 0.9279 | 0.9668 | 66.96% | 89.72% |
| M | 22,386 | 0.1317 | 0.0932 | 0.7173 | 0.8734 | 62.01% | 89.08% |
| X | 1,082 | 0.2181 | 0.1667 | −0.1556 | 0.5969 | 30.87% | 58.69% |

Regression metrics use `log10(flux)`; errors are in dex. Coverage counts
targets within 1 or 1.96 predicted standard deviations in the checkpoint's
normalized `log10(flux + 1e-8)` space.

Download the full summaries:

- {download}`Regression metrics <../model_assets/UncertaintyModelResults/evaluation/metrics/performance_metrics.csv>`
- {download}`Uncertainty metrics <../model_assets/UncertaintyModelResults/evaluation/metrics/uncertainty_metrics.csv>`

## Predicted versus measured flux

```{figure} ../model_assets/UncertaintyModelResults/evaluation/plots/regression_plot.png
:alt: Density of predicted versus measured SXR flux, with a perfect-agreement line and class-specific MAE band.
:width: 100%

Full-disk flux in W/m². The red line marks perfect agreement; shading shows
class-specific log-space MAE.
```

## Calibration

```{figure} ../model_assets/UncertaintyModelResults/evaluation/plots/uncertainty_calibration.png
:alt: Nominal versus empirical coverage by class and a histogram of standardized residuals.
:width: 100%

Nominal versus measured coverage (left) and standardized residuals (right).
```

To evaluate your own predictions, follow the [command-line workflow](workflows.md#evaluate-predictions).

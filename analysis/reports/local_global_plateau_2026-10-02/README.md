# Local/global plateau review

Read from local W&B history records. Only completed validation records are included; active runs can have newer checkpoints than flushed history.

| Run | Best epoch | Validation MSE | MAE at that epoch |
| --- | ---: | ---: | ---: |
| FOXES-OG-Uncertainty-Correction-Head | 10 | 0.0151221 | 0.0734174 |
| FOXES-OG-PATCH-GLOBAL-FULL-UNFROZEN | 3 | 0.0145142 | 0.0734029 |
| FOXES-OG-PATCH-GLOBAL-INDEPENDENT | 3 | 0.0150548 | 0.0770161 |

Metrics above use normalized log10(flux + 1e-8); they are not directly comparable to the standalone evaluator's log10(flux) metrics.

Independent run: training corrected MSE decreased from 0.0046412 at the best validation epoch to 0.0019245 at epoch 14, while validation MSE increased from 0.0150548 to 0.0178083.

At the independent run's best validation epoch, local MSE is 0.0151276; corrected MSE is 0.0150548, a 0.48% decrease. This does not establish a statistically reliable improvement.

M-class validation examples are 9.1% of the four logged classes and contribute 43.7% of their total squared log-flux error at that epoch.

Interpretation: the growing training/validation gap suggests generalization is a more useful next focus than architecture capacity or gradient coupling alone. This is not proof of an irreducible data limit.

Next diagnostic: evaluate the released checkpoint and the selected local/global checkpoints on identical validation timestamps and with identical metrics. Then inspect paired residuals by flare class, date, and active-region/event group. Keep the final test split for final assessment.

[Training/validation curves](training_validation.png)
[Epoch metrics](epoch_metrics.csv)
[Run summaries](summary.json)

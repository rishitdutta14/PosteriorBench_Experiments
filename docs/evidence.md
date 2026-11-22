# Evidence index

Each row links the central claim to the committed numerical evidence and its implementation. Directories under `results/` hold saved evidence; fresh runs write elsewhere.

| Experiment / claim | Numeric evidence | Implementation |
|---|---|---|
| 1 · spatial detection rates | [Rates](../results/01_spatial_dependence/spatial_metric_detection/results/detection_rates.csv), [all scores](../results/01_spatial_dependence/spatial_metric_detection/results/scores.csv), [null calibration](../results/01_spatial_dependence/spatial_metric_detection/results/calibration.json) | [Metric comparisons](../experiments/01_spatial_dependence/poisson_metric_detection.py) |
| 1 · controlled generator | [Fitted design](../results/01_spatial_dependence/spatial_metric_detection/generator/design.json), [reference correlations](../results/01_spatial_dependence/poisson_reference_correlations/summary.csv) | [Generator](../experiments/01_spatial_dependence/poisson_smooth_fields.py) |
| 2 · final 4×/512-member comparison and 100× injected-noise sweep | [Final settings and tables](../results/02_sensor_placement/final) | [Final-setting runner](../experiments/02_sensor_placement/presentation_analysis.py) |
| 3 · 45.02× paired variance, PCA modes | [Per case](../results/03_posterior_geometry/dimensions/per_case_100.csv), [paired ratios](../results/03_posterior_geometry/dimensions/paired_ratios.csv), [subsets](../results/03_posterior_geometry/dimensions/subsets_38.csv) | [Analysis](../experiments/03_posterior_geometry/esmda_paper_comparison/analyze_dimensions.py) |
| 3 · original solver settings | [Settings](../results/03_posterior_geometry/run/run_settings.json), [official evaluation](../results/03_posterior_geometry/official_evaluation.json) | [Generation](../experiments/03_posterior_geometry/esmda_paper_comparison/run_comparison.py) |
| 4 · paired pressure fit | [Pressure per case](../results/04_darcy_consistency/pressure_by_case.csv), [summary and bootstrap](../results/04_darcy_consistency/summary.json) | [Analysis](../experiments/04_darcy_consistency/darcy_pressure/analyze_pressures.py) |
| 4 · conservation and solver controls | [Conservation](../results/04_darcy_consistency/conservation_by_case.csv), [solver controls](../results/04_darcy_consistency/solver_controls_by_case.csv) | [Grid/flux implementation](../experiments/04_darcy_consistency/darcy_pressure/darcy_grid.py) |
| 5 · light-detail and layer decomposition | [Per case](../results/05_ltmi_recovery/posterior_analysis/per_case.csv), [summary](../results/05_ltmi_recovery/posterior_analysis/summary.json) | [Analysis](../experiments/05_ltmi_recovery/posterior_analysis/analyze.py) |

`tools/validate_results.py` recomputes headline comparisons from saved records. `tools/plot_results.py` builds five compact overview figures from those same tables. Original large field arrays are intentionally outside Git; complete generation commands are documented in [reproduction.md](reproduction.md).

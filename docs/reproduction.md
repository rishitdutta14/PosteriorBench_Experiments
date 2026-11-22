# Reproduction

[← Overview](../README.md) · [Evidence index](evidence.md)

There are three levels: inspect saved evidence, replot its tables, or regenerate ensembles and analyses. The first two require no training data. Raw HDF5 predictions and Arrow datasets are not committed.

## Inspect and replot

From the repository root:

```bash
python -m pip install -r requirements-figures.txt
python tools/validate_results.py
python tools/plot_results.py --output runs/summary_figures
```

The validator checks the 4,900 spatial comparison records, the exact sensor sweep through 100×, all 90 paired Poisson cases, all 90 Darcy pressure comparisons, the LTMI error decomposition, and local documentation links. It does not rerun inference. The plotting command creates five new compact figures; it does not overwrite archived presentation figures.

## Full environment and data

The original analyses ran on CPU using Python 3.13. The LTMI downloader uses `itertools.batched`, which requires Python 3.12 or newer; Python 3.13 is recommended for consistency.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

A focused upstream snapshot is included, so a separate PosteriorBench checkout is unnecessary. The launcher sets its import path. Download only the task you intend to run:

```bash
python tools/fetch_data.py poisson
python tools/fetch_data.py darcy
python tools/fetch_data.py ltmi
```

Downloads are pinned to dataset revision `f45719309a0ea6500b86e6e9973c18b3ffff8650`. Darcy and LTMI download their complete training pools; these are substantial downloads. Poisson downloads posterior evaluation inputs, normalization metadata, and only the first training shard used in this geometry run. The Poisson runner verifies that shard's SHA-256 before materialising it.

Existing data can be reused without copying it:

```bash
python tools/run_experiment.py 3 --data /absolute/path/to/data
```

The directory supplied to `--data` should contain `PosteriorDataset_hf/` and `PDEFieldDataset_hf/`. The same location can be supplied with the `POSTERIORBENCH_DATA` environment variable. The upstream dataset license continues to apply; data are downloaded from the original source.

Use `--dry-run` to inspect the exact commands before running them:

```bash
python tools/run_experiment.py 4 --dry-run
```

## 1. Spatial dependence

```bash
python tools/run_experiment.py 1
```

This measures weighted pixel and patch correlations, fits the Gaussian controls, then runs the 1,000 null and 300-per-variant metric comparisons. Fresh outputs go to `runs/01_spatial_dependence/`. The original scores and design are under `results/01_spatial_dependence/`.

`spatial_dependence.py` is the earlier patch-shuffle diagnostic and is retained as supporting code; the main reported experiment is the calibrated smooth-field sensitivity study. Large per-case correlation tables and fitted spectral arrays can be regenerated; the committed summary curves, fitted design, and all metric comparison scores support inspection of the results.

## 2. Sensor placement

```bash
python tools/run_experiment.py 2
```

No released data are needed: the study is synthetic and reads the committed sensor coordinates. The final runner uses **512 members and 4× assumed noise**, and sweeps **injected noise through 100×**. Outputs go to `runs/02_sensor_placement/`. A deterministic covariance-only run is also available:

```bash
python experiments/02_sensor_placement/presentation_analysis.py --sweep-only --output runs/sensor_exact_only
```

The original final map figure is archived. Its original export script was not saved; the new runner uses a fully recorded seed to replay the final setting. Its stochastic map pixels may differ. Exact covariance results for the saved layouts are deterministic. This distinction is recorded rather than presented as bitwise reproduction of the original export.

## 3. Poisson posterior geometry

```bash
python tools/run_experiment.py 3
```

This generates 90 ensembles from the 3,572-field prior shard, evaluates them with the released scorer, analyses 100-map and 38-map ensembles, and redraws the case-12 moment/PCA figures. Generation writes to `experiments/03_posterior_geometry/esmda_paper_comparison/run/`; geometry results go to its `dimensions/` directory. These generated files are ignored by Git. The committed evidence is under `results/03_posterior_geometry/`.

The generic `run_experiment.py` inside this experiment directory supplies shared weighted-PCA and nonlinear-scale analysis functions. It also contains the earlier pilot CLI. The supported final reproduction route is the top-level launcher above, which selects the released-profile comparison.

## 4. Darcy consistency

```bash
python tools/run_experiment.py 4
```

The launcher generates aligned permeability/internal-pressure fields, independently solves the saved rounded permeability, computes pressure and conservation diagnostics, and plots the results. The scripts resume validated predictions and numerical-pressure files. They check shape, finiteness, case pairing, and permeability hashes.

Generation writes to `experiments/04_darcy_consistency/darcy_esmda/run/`; numerical pressures and fresh analysis outputs go under `darcy_pressure/`. These are separate from the committed `results/04_darcy_consistency/` evidence.

## 5. LTMI recovery

```bash
python tools/run_experiment.py 5
```

This runs the released 1,024-member, one-update configuration, saves 100 aligned material/light draws per case, evaluates the layer average, and performs the fine-detail and individual-layer analyses. Generated predictions, evaluation outputs, and analysis outputs remain under `experiments/05_ltmi_recovery/`, outside the archived evidence.

## Reusing original large ensembles

For studies 3–5, `--skip-generation` runs the analysis on existing predictions at the generated-output paths documented above. The source machine's original HDF5 files are not needed to inspect this release, and are not published by these scripts. To reanalyse them locally, copy or link the relevant prediction directories into the documented paths first; include the run settings and evaluator JSON for the Poisson geometry analysis.

## What has been verified for this release?

See [`provenance/PACKAGING.md`](../provenance/PACKAGING.md) for the actual validation performed during packaging. Full fresh 90-case generation across all tasks has not been repeated merely to prepare the repository. Runtime estimates are intentionally not invented: the largest costs are the original datasets, joint-ensemble generation, and 9,000 Darcy solves. Use the data-free summary route for an immediate check.

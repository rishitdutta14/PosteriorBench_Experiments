# Packaging and validation notes

## Authorship and source

The original benchmark source comes from `neuraloperator/PosteriorBench` at commit `ab416f5b322717f5668de01889ae0141dbc00af6`. The local upstream checkout was clean when inspected. Included upstream source files are unmodified and retain their MIT license. Only the modules and profiles needed by this release are included; this is not the complete upstream project.

Experiment scripts and archived results come from Rishit Dutta's existing local experiments. Portable paths and a pinned source identifier replace machine-specific paths and dependence on a nested Git checkout. Original and packaged file hashes are listed in `source_files.json`; these hashes document packaging, not a rerun of the scientific experiments. Machine-local paths in archived JSON metadata are elided while numerical settings and results are retained.

## Final experiment choices

- Sensor-placement interpretation uses the final **4× assumed-noise, 512-member** comparison and the **injected-noise sweep through 100×**. Earlier 100× solver-profile tables are not the reported sensor experiment.
- Poisson geometry uses `esmda_paper_comparison`, with the verified 3,572-field shard and released 100× solver noise profile. Its 45.02× variance / 10-versus-14 PCA result supersedes the older pilot account for this release.
- Darcy and LTMI use their saved complete released-prior-pool runs.

The original sensor map-export script and 100× sweep-extension export script were not saved. A new explicit final-setting runner was added. It uses the saved coordinates, computes exact Gaussian covariance through 100×, and offers a documented-seed 4×/512-member replay. The archived presentation map is retained. The covariance sweep is deterministic; the new stochastic replay is not claimed to reproduce the original image bit for bit.

## Validation

The lightweight validator recomputes detection counts from the 4,900 comparison scores, paired PCA ratios from the 90-case table, Darcy comparisons from all case records, and the pooled LTMI decomposition from per-case RMSEs. It also checks local Markdown file links. Summary figures can be regenerated directly from the committed tables.

Packaging checks additionally compile the Python source and exercise documented command-line entry points. The deterministic final sensor sweep is recomputed. The exact set of executed checks and their outcome is recorded in `validation.json` after verification.

Full fresh 90-case Poisson, Darcy and LTMI generation has not been repeated during packaging. The original run outputs are the evidence for those results; this release supplies the generation/analysis code and pinned inputs needed for a subsequent reproduction.

## Excluded files

Training data, raw posterior HDF5s, model weights, caches, local environments, browser screenshots, and machine logs are excluded from Git. Data can be obtained at the pinned original source. No authentication material is required in the repository.

The upstream license covers the upstream code. No new license has been assigned to the original experimental additions in this initial release.

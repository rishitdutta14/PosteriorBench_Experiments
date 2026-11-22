"""Fit smooth Gaussian controls to the published Poisson correlation curves."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "vendor" / "PosteriorBench"))

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.optimize import Bounds, LinearConstraint, brentq, minimize, nnls
from scipy.special import expit


FIELD_SIZE = 128
FFT_SIZE = 256
PATCH_SIZES = (1, 4, 8, 16)
DIRECTIONS = ("horizontal", "vertical")
PATTERNS = ("range", "direction", "anticorrelation")
FREQUENCY = np.fft.fftfreq(FFT_SIZE)
KY, KX = np.meshgrid(FREQUENCY, FREQUENCY, indexing="ij")
RADIUS_SQUARED = KX**2 + KY**2
RADIUS = FIELD_SIZE * np.sqrt(RADIUS_SQUARED)
RING_CENTERS = np.arange(0.5, 12.5, 0.5)
RING_WIDTHS = np.maximum(0.32, 0.09 * RING_CENTERS)
TAIL_MASS = 0.004
LAGS = np.arange(FIELD_SIZE)


def read_reference_curves(path):
    groups = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["split"], int(row["window_size_pixels"]), row["direction"],
                   int(row["center_distance_pixels"]))
            groups[key].append(float(row["mean_correlation"]))
    return {key: np.quantile(values, [0.1, 0.5, 0.9]) for key, values in groups.items()}


def ring_basis(scale=1.0):
    radius = scale * RADIUS
    rings = np.exp(-0.5 * ((radius[None] - RING_CENTERS[:, None, None])
                          / RING_WIDTHS[:, None, None]) ** 2)
    # A small, smooth spectral tail retains the prior's fine-scale variation.
    tail = (1 + radius**2) ** -2.25 * expit((radius - 6) / 0.8)
    spectra = np.concatenate([rings, tail[None]], axis=0)
    spectra[:, 0, 0] = 0.0
    return spectra / spectra.sum(axis=(1, 2), keepdims=True)


def box_power(size):
    # Squared transfer function of the stride-one square averaging window.
    axis_power = (np.sinc(size * FREQUENCY) / np.sinc(FREQUENCY)) ** 2
    return axis_power[:, None] * axis_power[None, :]


def directional_covariance(spectra, size, direction):
    perpendicular_axis = -2 if direction == "horizontal" else -1
    axis_spectrum = np.sum(spectra * box_power(size), axis=perpendicular_axis)
    return np.fft.ifft(axis_spectrum, axis=-1).real[..., :FIELD_SIZE] * FFT_SIZE


def correlation_curves(spectrum):
    curves = {}
    for size in PATCH_SIZES:
        for direction in DIRECTIONS:
            covariance = directional_covariance(spectrum, size, direction)
            curves[size, direction] = covariance / covariance[0]
    return curves


def crossing(curve, level=0.5):
    """Linearly interpolate the first downward crossing between integer lags."""
    index = int(np.flatnonzero(curve < level)[0])
    before, after = curve[index - 1:index + 1]
    return float(index - 1 + (before - level) / (before - after))


def fit_baseline(reference, basis):
    systems = []
    for size in PATCH_SIZES:
        covariance = directional_covariance(basis, size, "horizontal")
        distances = np.arange(size, FIELD_SIZE - size + 1)
        target = np.array([
            np.mean([reference["validation", size, direction, lag][1] for direction in DIRECTIONS])
            for lag in distances
        ])
        emphasis = np.where(distances <= 65, 1.0, 0.28)
        emphasis[distances <= 8] = 1.8
        if size == 1:
            emphasis[distances <= 3] = 8.0
        if size == 16:
            # The largest windows have fewer observed lags and lower variance in the fit.
            emphasis *= 1.7
        systems.append((covariance[:, distances] - covariance[:, :1] * target) * emphasis)

    n_rings = len(basis)
    system = np.vstack([*(matrix.T for matrix in systems), 0.02 * np.eye(n_rings),
                        np.full((1, n_rings), 50.0)])
    target = -system[:, -1] * TAIL_MASS
    target[-1] += 50.0
    ring_weights, _ = nnls(system[:, :-1], target)
    return np.r_[ring_weights / ring_weights.sum() * (1 - TAIL_MASS), TAIL_MASS]


def mix_spectrum(ring_weights, basis):
    return np.einsum("k,kij->ij", ring_weights, basis)


def range_spectrum(ring_weights, desired_crossing):
    def difference(scale):
        spectrum = mix_spectrum(ring_weights, ring_basis(scale))
        return crossing(correlation_curves(spectrum)[1, "horizontal"]) - desired_crossing

    scale = brentq(difference, 0.55, 1.65, xtol=1e-6)
    return mix_spectrum(ring_weights, ring_basis(scale))


def direction_spectrum(baseline, desired_gap):
    angle = np.divide(KX**2 - KY**2, RADIUS_SQUARED, out=np.zeros_like(RADIUS),
                      where=RADIUS_SQUARED > 0)
    # Swapping axes negates the modulation, so each radial shell keeps its mass.
    modulation = np.tanh(4 * angle)

    def difference(strength):
        curves = correlation_curves(baseline * (1 + strength * modulation))
        gap = abs(crossing(curves[1, "horizontal"]) - crossing(curves[1, "vertical"]))
        return gap - desired_gap

    strength = brentq(difference, 0.0, 1.0, xtol=1e-7)
    return baseline * (1 + strength * modulation)


def anticorrelation_spectrum(ring_weights, basis, desired_at_50):
    covariance = directional_covariance(basis, 1, "horizontal").T
    baseline_curve = covariance @ ring_weights
    adjustable = np.flatnonzero(RING_CENTERS <= 6)
    response = covariance[:, adjustable]
    initial = ring_weights[adjustable]
    delta = desired_at_50 - baseline_curve[50]
    target = delta * np.exp(-0.5 * ((LAGS - 50) / 11) ** 2)
    emphasis = np.where(LAGS <= 70, 1., 0.3)
    residual = response * emphasis[:, None]
    expected = target * emphasis
    equality = LinearConstraint(response[[0, 50]], [0., delta], [0., delta])
    tolerance = np.minimum(0.006, 0.04 * (1 - baseline_curve[1:21]))
    short_range = LinearConstraint(response[1:21], -tolerance, tolerance)

    def objective(shift):
        error = residual @ shift - expected
        value = error @ error + 0.1 * (shift @ shift)
        gradient = 2 * (residual.T @ error + 0.1 * shift)
        return value, gradient

    result = minimize(objective, np.zeros_like(initial), jac=True, method="SLSQP",
                      bounds=Bounds(-initial, 1. - initial), constraints=[equality, short_range],
                      options={"ftol": 1e-11, "maxiter": 500})
    if not result.success:
        raise ValueError(f"Could not match correlation {desired_at_50} at 50 pixels: {result.message}")
    weights = ring_weights.copy()
    weights[adjustable] += result.x
    return mix_spectrum(weights, basis)


def sampling_amplitude(spectrum, sigma):
    return np.asarray(sigma * FFT_SIZE * np.sqrt(spectrum[:, :FFT_SIZE // 2 + 1]), dtype=np.float32)


def draw_fields(amplitude, rng, count=100):
    """Filter white noise on a padded grid, then take a contiguous 128-pixel crop."""
    noise = rng.standard_normal((count, FFT_SIZE, FFT_SIZE), dtype=np.float32)
    coefficients = rfft2(noise, axes=(-2, -1), workers=1)
    coefficients *= amplitude
    fields = irfft2(coefficients, s=(FFT_SIZE, FFT_SIZE), axes=(-2, -1), workers=1,
                    overwrite_x=True)
    return np.ascontiguousarray(fields[:, :FIELD_SIZE, :FIELD_SIZE])


def describe_spectrum(name, pattern, target, spectrum, baseline_curves):
    np.testing.assert_allclose(spectrum.sum(), 1., atol=2e-10, rtol=0)
    if np.any(spectrum < 0):
        raise ValueError(f"Negative power in {name}")
    curves = correlation_curves(spectrum)
    horizontal = curves[1, "horizontal"]
    vertical = curves[1, "vertical"]
    range_x = crossing(horizontal)
    range_y = crossing(vertical)
    baseline_crossing = crossing(baseline_curves[1, "horizontal"])
    at_50 = float((horizontal[50] + vertical[50]) / 2)
    delta_50 = at_50 - baseline_curves[1, "horizontal"][50]
    short_change = max(float(np.max(np.abs(curves[1, axis][1:21] - baseline_curves[1, axis][1:21])))
                       for axis in DIRECTIONS)
    strength = {"range": (range_x + range_y) / 2 - baseline_crossing,
                "direction": abs(range_x - range_y), "anticorrelation": delta_50}.get(pattern, 0.)
    return {
        "name": name, "pattern": pattern, "target": target, "strength": float(strength),
        "horizontal_crossing_pixels": range_x, "vertical_crossing_pixels": range_y,
        "correlation_at_50": at_50, "max_correlation_change_through_20": short_change,
    }, curves


def check_baseline_fit(reference, baseline_curves):
    results = []
    for split in ("validation", "test"):
        for (size, direction), curve in baseline_curves.items():
            distances = np.arange(size, 65)
            median = np.array([reference[split, size, direction, lag][1] for lag in distances])
            rmse = float(np.sqrt(np.mean((curve[distances] - median) ** 2)))
            results.append({"split": split, "window_size": size, "direction": direction,
                            "rmse_through_64": rmse})
            if rmse > 0.035:
                raise ValueError(f"Baseline correlation RMSE {rmse:.3f} in {split}, {size}px, {direction}")
    return results


def save_fit_plots(output, reference, variants, model_curves, spectra, sigma):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharey=True)
    for size, axis in zip(PATCH_SIZES, axes.flat):
        lags = np.arange(size, 81)
        stats = np.array([reference["test", size, "horizontal", lag] for lag in lags])
        axis.fill_between(lags, stats[:, 0], stats[:, 2], color="#9bd0d2", alpha=0.6,
                          label="Poisson test cases, middle 80%")
        axis.plot(lags, stats[:, 1], color="#276677", label="Poisson test median")
        axis.plot(lags, model_curves["baseline"][size, "horizontal"][lags], "--", color="#8a2961",
                  label="Smooth baseline")
        axis.axhline(0., color="0.75", linewidth=0.7)
        axis.set_title(f"{size} × {size} pixels", loc="left")
    axes.flat[0].legend(frameon=False)
    figure.supxlabel("Center-to-center distance (pixels)")
    figure.supylabel("Correlation")
    figure.suptitle("Smooth-field baseline and published Poisson correlation curves")
    figure.savefig(output / "baseline_fit.png", dpi=170)

    figure = Figure(figsize=(9, 8.2), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2, sharex=True, sharey=True)
    for row, pattern in enumerate(PATTERNS):
        choices = [item for item in variants if item["pattern"] == pattern]
        for axis, size in zip(axes[row], (1, 16)):
            lags = np.arange(size, 71)
            axis.plot(lags, model_curves["baseline"][size, "horizontal"][lags], color="0.25",
                      linestyle=":", linewidth=2.1, label="Baseline")
            for index, item in enumerate(choices):
                color = f"C{index}"
                curve = model_curves[item["name"]]
                axis.plot(lags, curve[size, "horizontal"][lags], color=color, label=item["name"])
                if pattern == "direction":
                    axis.plot(lags, curve[size, "vertical"][lags], color=color, linestyle="--")
            axis.axhline(0., color="0.75", linewidth=0.7)
            label = {"range": "Dependence range", "direction": "Direction",
                     "anticorrelation": "Long-range anticorrelation"}[pattern]
            axis.set_title(f"{label} · {size} × {size} pixels", loc="left")
        axes[row, 1].legend(frameon=False, fontsize=7, loc="upper right")
    figure.supxlabel("Center-to-center distance (pixels)")
    figure.supylabel("Correlation")
    figure.suptitle("Perturbations of the smooth-field baseline\n"
                    "Directional curves show horizontal (solid) and vertical (dashed)")
    figure.savefig(output / "perturbation_curves.png", dpi=170)

    figure = Figure(figsize=(10, 3), layout="constrained")
    FigureCanvasAgg(figure)
    names = ["baseline", "range -4 px", "direction 8 px", "correlation -0.10"]
    for name, axis in zip(names, figure.subplots(1, len(names))):
        example = draw_fields(sampling_amplitude(spectra[name], sigma), np.random.default_rng(1247), 1)[0]
        image = axis.imshow(example, cmap="RdBu_r", vmin=-3 * sigma, vmax=3 * sigma)
        axis.set_title(name.capitalize())
        axis.set_axis_off()
    figure.colorbar(image, ax=figure.axes, shrink=0.7, label="Source field value")
    figure.suptitle("Example fields using the same underlying random draw")
    figure.savefig(output / "field_examples.png", dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-curves", type=Path,
                        default=Path("outputs/poisson_patch_correlations/per_case.csv"))
    parser.add_argument("--training-metadata", type=Path,
                        default=Path("data/PDEFieldDataset_hf/Poisson/metadata.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    reference = read_reference_curves(args.reference_curves)
    metadata = json.loads(args.training_metadata.read_text())
    field = next(name for name, channels in metadata["field_channels"].items() if channels == ["f"])
    index = metadata["field_names"].index(field)
    sigma = metadata["stats"]["std"][index]
    basis = ring_basis()
    ring_weights = fit_baseline(reference, basis)
    baseline = mix_spectrum(ring_weights, basis)
    baseline_curves = correlation_curves(baseline)
    fit_results = check_baseline_fit(reference, baseline_curves)
    spectra = {"baseline": baseline}
    model_curves = {}
    configs = [{"name": "baseline", "pattern": "baseline", "target": 0}]
    base_range = crossing(baseline_curves[1, "horizontal"])
    for change in (-4, -2, -1, 1, 2, 4):
        name = f"range {change:+} px"
        spectra[name] = range_spectrum(ring_weights, base_range + change)
        configs.append({"name": name, "pattern": "range", "target": change})
    for gap in (1, 2, 4, 8):
        name = f"direction {gap} px"
        spectra[name] = direction_spectrum(baseline, gap)
        np.testing.assert_allclose(spectra[name] + spectra[name].T, 2 * baseline, atol=1e-14)
        configs.append({"name": name, "pattern": "direction", "target": gap})
    for target in (-0.18, -0.15, -0.10):
        name = f"correlation {target:.2f}"
        spectra[name] = anticorrelation_spectrum(ring_weights, basis, target)
        configs.append({"name": name, "pattern": "anticorrelation", "target": target})

    variants = []
    for config in configs:
        name = config["name"]
        description, curves = describe_spectrum(**config, spectrum=spectra[name],
                                                baseline_curves=baseline_curves)
        if config["pattern"] == "anticorrelation":
            np.testing.assert_allclose(description["correlation_at_50"], config["target"], atol=1e-7)
            if description["max_correlation_change_through_20"] > 0.0060001:
                raise ValueError(f"{name} also changes short-range correlation too much")
        variants.append(description)
        model_curves[name] = curves
        print(name, f'crossings {description["horizontal_crossing_pixels"]:.2f} / '
              f'{description["vertical_crossing_pixels"]:.2f}, '
              f'correlation at 50 px {description["correlation_at_50"]:+.3f}', flush=True)

    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "spectra.npz", **spectra)
    report = {
        "field_size": FIELD_SIZE, "padded_fft_size": FFT_SIZE, "marginal_mean": 0.,
        "marginal_std": sigma, "baseline_spectral_tail_mass": TAIL_MASS,
        "swd_normalization_mean": metadata["stats"]["mean"][index],
        "swd_normalization_std": sigma, "baseline_fit": fit_results, "variants": variants,
    }
    (args.output / "design.json").write_text(json.dumps(report, indent=2) + "\n")
    save_fit_plots(args.output, reference, variants, model_curves, spectra, sigma)


if __name__ == "__main__":
    main()

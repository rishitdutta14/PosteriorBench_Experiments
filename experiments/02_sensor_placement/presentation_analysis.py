"""Final sensor study: 4x assumed ES-MDA noise; exact noise sweep through 100x.

The original slide image is archived in figures/. This explicit runner uses the
saved sensor coordinates and a documented seed to regenerate the final setting.
Stochastic map pixels need not be bit-identical to the original slide export.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn
from threadpoolctl import threadpool_limits

from run_ablation import Conditioner, NOISE, Poisson, SIZE, at_points, prior_std, rms, write_csv

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT / 'results' / '02_sensor_placement'
NAMES = ('even', 'diagonal', 'one_cluster', 'four_clusters')
LABELS = {'even': 'Evenly spread', 'diagonal': 'Main diagonal',
          'one_cluster': 'One cluster', 'four_clusters': 'Four clusters'}
MULTIPLIERS = (0.5, 1., 4., 16., 64., 100.)


def load_coordinates():
    layouts = {}
    for folder in ('paper_esmda_settings', 'cluster_ablation'):
        with np.load(ARCHIVE / folder / 'coordinates.npz') as saved:
            layouts.update({name: saved[name] for name in NAMES if name in saved})
    assert all(layouts[name].shape == (3, 128, 2) for name in NAMES)
    return layouts


def main(output, sweep_only=False):
    output.mkdir(parents=True, exist_ok=False)
    layouts = load_coordinates()
    std = prior_std()
    basis = dctn(np.eye(SIZE), norm='ortho', axes=(0,))
    prior_var = ((basis.T ** 2) @ std ** 2 @ basis ** 2).ravel()
    prior_scale = float(np.sqrt(prior_var.mean()))
    model = Poisson()
    rng = np.random.default_rng(20261120)
    # One common 512-member bank and common injected-noise values, across layouts.
    prior = idctn(rng.standard_normal((512, SIZE, SIZE)) * std,
                  norm='ortho', axes=(-2, -1)) if not sweep_only else None
    potential = model.forward(prior) if prior is not None else None
    prior_empirical_std = prior.std(axis=0) if prior is not None else None
    noise = rng.standard_normal((512, 128)) if prior is not None else None
    rows, esmda_rows, maps = [], [], {}
    for name in NAMES:
        for repeat, coords in enumerate(layouts[name]):
            design = model.sensor_design(coords, std)
            cross = idctn(design.reshape(-1, SIZE, SIZE) * std,
                           norm='ortho', axes=(-2, -1)).reshape(128, -1)
            cov = design @ design.T
            for multiplier in MULTIPLIERS:
                exact = Conditioner(cov, cross, np.empty((128, 0)), NOISE * multiplier)
                field_std = exact.posterior_std(prior_var, cross).reshape(SIZE, SIZE)
                rows.append(dict(layout=name, repeat=repeat, injected_noise_multiplier=multiplier,
                                 source_rms_std_prior_ratio=rms(field_std) / prior_scale))
                if multiplier == 1 and repeat == 0:
                    maps[name, 'reference'] = field_std / np.sqrt(prior_var.reshape(SIZE, SIZE))
            if prior is not None and repeat == 0:
                readings = at_points(potential, coords)
                anom = readings - readings.mean(axis=0)
                source = prior.reshape(512, -1)
                empirical_cov = anom.T @ anom / 511
                empirical_cross = anom.T @ (source - source.mean(axis=0)) / 511
                for scale in (4., 16., 64.):
                    sigma = np.maximum(1e-6, scale * np.sqrt(np.diag(empirical_cov)))
                    empirical = Conditioner(empirical_cov, empirical_cross, np.empty((128, 0)),
                                             sigma, jitter=1e-8)
                    # The common observed vector only shifts the mean; it cannot
                    # change the ensemble spread in this linear one-update test.
                    updated, _ = empirical.update(prior, np.empty((512, 0)),
                                                   noise * sigma - readings)
                    ratio = updated.std(axis=0) / prior_empirical_std
                    esmda_rows.append(dict(layout=name, assumed_noise_multiplier=scale,
                                           members=512, retained_rms_source_spread=rms(updated.std(axis=0))/rms(prior_empirical_std)))
                    if scale == 4:
                        maps[name, 'esmda'] = ratio
            print(f'{name}: placement {repeat + 1}/3', flush=True)
    write_csv(output / 'exact_noise_sweep.csv', rows)
    if esmda_rows:
        write_csv(output / 'esmda_retained_spread.csv', esmda_rows)
    (output / 'settings.json').write_text(json.dumps(dict(
        seed=20261120, source_prior={'alpha': 2.25, 'tau': 3.0}, sensors=128,
        grid=128, prior_members=512, one_update=True, assumed_noise_scale=4,
        assumed_noise_definition='4 times initial predicted sensor standard deviation; floor 1e-6; jitter 1e-8',
        actual_base_noise_std=NOISE, injected_noise_multipliers=MULTIPLIERS,
        note='Exact covariance sweep is independent of observed values for a fixed linear Gaussian prior. Stochastic comparison uses a documented fresh replay seed; original slide export retained separately.'
    ), indent=2) + '\n')
    plot(rows, maps, layouts, output)


def plot(rows, maps, layouts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#24609a', '#925379', '#ae742b', '#27816c')
    fig, ax = plt.subplots(figsize=(8.7, 4.6), layout='constrained')
    for name, color in zip(NAMES, colors):
        values = [np.mean([r['source_rms_std_prior_ratio'] for r in rows
                           if r['layout']==name and r['injected_noise_multiplier']==m]) for m in MULTIPLIERS]
        ax.plot(MULTIPLIERS, 100*np.array(values), 'o-', label=LABELS[name], color=color)
    ax.set(xscale='log', xlabel='Injected measurement-noise standard deviation / base noise',
           ylabel='Source RMS posterior standard deviation (% of prior)',
           title='Sensor informativeness depends on the noise level')
    ax.set_xticks(MULTIPLIERS, [f'{m:g}x' for m in MULTIPLIERS])
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'noise_sweep_to_100x.png', dpi=180)
    plt.close(fig)
    if not any(key[1]=='esmda' for key in maps): return
    fig, axes = plt.subplots(2,4,figsize=(12,6.1), layout='constrained')
    for col,name in enumerate(NAMES):
        for row,method in enumerate(('reference','esmda')):
            image=axes[row,col].imshow(maps[name,method],vmin=0,vmax=1,cmap='viridis')
            coords=layouts[name][0]
            axes[row,col].scatter(coords[:,1],coords[:,0],s=2,c='white')
            axes[row,col].set(xticks=[],yticks=[])
            if row==0: axes[row,col].set_title(LABELS[name])
    axes[0,0].set_ylabel('Exact posterior / exact prior')
    axes[1,0].set_ylabel('ES-MDA after / before update')
    fig.colorbar(image,ax=axes,label='Pixelwise standard deviation ratio',shrink=.75)
    fig.suptitle('4x assumed sensor noise; 512 ES-MDA members; actual reference noise fixed')
    fig.savefig(output/'sensor_layouts_4x_512.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'runs'/'02_sensor_placement')
    parser.add_argument('--sweep-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=3): main(args.output,args.sweep_only)

"""Regenerate compact figures from committed numerical evidence; no inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
COLORS = ['#245a91', '#b65031', '#27816c', '#8b4b76', '#a18124']


def read(path):
    with path.open(encoding='utf-8') as stream: return list(csv.DictReader(stream))


def save(fig, output, name):
    fig.savefig(output / name, dpi=180)
    plt.close(fig)


def plot(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                         'axes.spines.right':False, 'font.size':10})
    rates = read(RESULTS/'01_spatial_dependence/spatial_metric_detection/results/detection_rates.csv')
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
    metrics = [('mean_rel_l2','Mean'),('std_rel_l2','Std'),('mmd','MMD'),('swd','SWD'),('spectral_rel_geomean','Spectral')]
    for ax,family in zip(axes,('shorter range','longer range')):
        for (metric,label),color in zip(metrics,COLORS):
            rows=sorted((r for r in rates if r['family']==family and r['metric']==metric),key=lambda r:float(r['strength']))
            ax.plot([float(r['strength']) for r in rows],[100*float(r['rate']) for r in rows],'o-',label=label,color=color)
        ax.axhline(80,color='#666',ls='--',lw=.8)
        ax.set(title=family.capitalize(),xlabel='Correlation-range shift magnitude (pixels)',ylim=(0,104))
    axes[0].set_ylabel('Detection rate (%)')
    axes[1].legend(frameon=False)
    save(fig,output,'01_range_detection.png')

    rows=read(RESULTS/'02_sensor_placement/final/exact_noise_sweep.csv')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for name,label,color in zip(('even','diagonal','one_cluster','four_clusters'),('Evenly spread','Main diagonal','One cluster','Four clusters'),COLORS):
        levels=sorted({float(r['injected_noise_multiplier']) for r in rows})
        ax.plot(levels,[100*np.mean([float(r['source_rms_std_prior_ratio']) for r in rows if r['layout']==name and float(r['injected_noise_multiplier'])==m]) for m in levels],'o-',label=label,color=color)
    ax.set(xscale='log',xlabel='Injected noise standard deviation / base noise',ylabel='Source spread (% of prior)',title='Exact posterior uncertainty through 100x injected noise')
    ax.set_xticks(levels,[f'{m:g}x' for m in levels])
    ax.legend(frameon=False)
    save(fig,output,'02_noise_sweep.png')

    pca=read(RESULTS/'03_posterior_geometry/dimensions/per_case_100.csv')
    bycase={}
    for r in pca: bycase.setdefault(r['case'],{})[r['method']]=r
    ratios=np.array([float(x['esmda']['variance_per_pixel'])/float(x['reference']['variance_per_pixel']) for x in bycase.values()])
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    axes[0].hist(ratios,bins=16,color=COLORS[1])
    axes[0].axvline(np.median(ratios),color='#333',ls='--')
    axes[0].set(xlabel='ES-MDA / reference variance',ylabel='Cases',title=f'Median paired ratio: {np.median(ratios):.2f}x')
    axes[1].boxplot([[float(r['d90']) for r in pca if r['method']==method] for method in ('reference','esmda')],tick_labels=['Reference','ES-MDA'])
    axes[1].set(ylabel='PCA modes for 90% variance',title='Concentration of the variance spectrum')
    save(fig,output,'03_posterior_geometry.png')

    pressure=read(RESULTS/'04_darcy_consistency/pressure_by_case.csv')
    paired={}
    for r in pressure:
        if r['region']=='held_out' and r['ensemble'] in ('internal','numerical'):
            paired.setdefault(r['case_id'],{})[r['ensemble']]=float(r['truth_mean_rmse'])*1000
    fig,ax=plt.subplots(figsize=(5.5,4.7),layout='constrained')
    x=np.array([r['internal'] for r in paired.values()]); y=np.array([r['numerical'] for r in paired.values()])
    lo,hi=min(x.min(),y.min())*.95,max(x.max(),y.max())*1.05
    ax.plot([lo,hi],[lo,hi],'--',color='#777')
    ax.scatter(x,y,color=COLORS[1],s=23,alpha=.8)
    ax.set(xlabel='Internal-pressure mean RMSE (x10^-3)',ylabel='Re-solved-pressure mean RMSE (x10^-3)',title='Re-solving worsens held-out pressure fit in 90/90 cases',xlim=(lo,hi),ylim=(lo,hi))
    save(fig,output,'04_darcy_pressure.png')

    report=json.loads((RESULTS/'05_ltmi_recovery/posterior_analysis/summary.json').read_text())
    fields=('average','layer_1','layer_2','half_difference')
    errors=[100*report['metrics'][f'material_{name}_mean_relative_error']['mean'] for name in fields]
    fig,ax=plt.subplots(figsize=(7.5,4),layout='constrained')
    bars=ax.bar(['Average','Layer 1','Layer 2','Half-difference'],errors,color=[COLORS[0],COLORS[1],COLORS[2],COLORS[3]])
    ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set(ylabel='Relative posterior mean-map error (%)',ylim=(0,max(errors)*1.2),title='The layer average conceals opposing mean errors')
    save(fig,output,'05_ltmi_layers.png')
    print(f'Wrote five figures to {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/summary_figures')
    plot(parser.parse_args().output)

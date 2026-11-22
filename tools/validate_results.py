"""Check headline claims against per-case records, plus local documentation links."""
import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'


def read(relative):
    with (RESULTS/relative).open(encoding='utf-8') as f: return list(csv.DictReader(f))


def main():
    rates=read('01_spatial_dependence/spatial_metric_detection/results/detection_rates.csv')
    scores=read('01_spatial_dependence/spatial_metric_detection/results/scores.csv')
    assert len(scores)==4900
    config=json.loads((RESULTS/'01_spatial_dependence/spatial_metric_detection/results/calibration.json').read_text())
    groups={}
    for r in scores: groups.setdefault(r['variant'],[]).append(r)
    for r in rates:
        observed=sum(float(x[r['metric']])>config['cutoffs'][r['metric']] for x in groups[r['variant']])
        assert observed==int(r['detections']) and len(groups[r['variant']])==int(r['comparisons'])
        assert math.isclose(observed/int(r['comparisons']),float(r['rate']),abs_tol=1e-14)
    assert next(float(r['rate']) for r in rates if r['variant']=='range -4 px' and r['metric']=='swd')==280/300
    assert next(float(r['rate']) for r in rates if r['variant']=='range +4 px' and r['metric']=='swd')==1
    assert round(100*max(float(r['rate']) for r in rates if r['family']=='direction'))==19
    assert round(100*max(float(r['rate']) for r in rates if r['family']=='anticorrelation'))==15
    print('1: calibration, 4,900 scores and all detection rates agree.')

    sweep=read('02_sensor_placement/final/exact_noise_sweep.csv')
    levels=sorted({float(r['injected_noise_multiplier']) for r in sweep})
    assert levels==[.5,1.,4.,16.,64.,100.] and len(sweep)==72
    for level in levels:
        means={name:statistics.mean(float(r['source_rms_std_prior_ratio']) for r in sweep if r['layout']==name and float(r['injected_noise_multiplier'])==level) for name in ['even','diagonal','one_cluster','four_clusters']}
        winner=min(means,key=means.get)
        assert winner==('even' if level<=16 else 'four_clusters'), (level,means)
    print('2: all 72 exact-covariance records and the layout reversal through 100x agree.')

    pca=read('03_posterior_geometry/dimensions/per_case_100.csv')
    pairs={}
    for r in pca: pairs.setdefault(r['case'],{})[r['method']]=r
    assert len(pairs)==90 and len(pca)==180
    ratios=[float(r['esmda']['variance_per_pixel'])/float(r['reference']['variance_per_pixel']) for r in pairs.values()]
    assert all(x>1 for x in ratios) and round(statistics.median(ratios),2)==45.02
    for method,expected in [('reference',14),('esmda',10)]:
        assert statistics.median(float(r['d90']) for r in pca if r['method']==method)==expected
    assert all(float(r['esmda']['participation_ratio'])<float(r['reference']['participation_ratio']) for r in pairs.values())
    print('3: 90 paired PCA cases confirm 45.02x variance and 10 versus 14 modes.')

    pressure=read('04_darcy_consistency/pressure_by_case.csv')
    summary=json.loads((RESULTS/'04_darcy_consistency/summary.json').read_text())
    cases={}
    for r in pressure:
        if r['region']=='held_out': cases.setdefault(r['case_id'],{})[r['ensemble']]=r
    assert len(cases)==90
    for metric in ['truth_mean_rmse','reference_w1','truth_crps']:
        assert all(float(x['numerical'][metric])>float(x['internal'][metric]) for x in cases.values())
        for method in ['internal','numerical','reference']:
            mean=statistics.mean(float(x[method][metric]) for x in cases.values())
            assert math.isclose(mean,summary['pressure'][method]['held_out'][metric],rel_tol=1e-12,abs_tol=1e-14)
    assert summary['solver_controls']['max_truth_resolve_rmse']<2e-10
    print('4: all 90 pressure comparisons and archived summary averages agree.')

    rows=read('05_ltmi_recovery/posterior_analysis/per_case.csv')
    summary=json.loads((RESULTS/'05_ltmi_recovery/posterior_analysis/summary.json').read_text())
    assert len(rows)==90 and summary['cases']==90
    for field in ['average','layer_1','layer_2','half_difference']:
        key=f'material_{field}_mean_relative_error'
        assert math.isclose(statistics.mean(float(r[key]) for r in rows),summary['metrics'][key]['mean'],rel_tol=1e-12)
    # Recover the pooled decomposition from per-case RMSEs rather than averaging fractions.
    common=sum(float(r['material_average_mean_rmse'])**2 for r in rows)
    contrast=sum(float(r['material_half_difference_mean_rmse'])**2 for r in rows)
    assert math.isclose(contrast/(common+contrast),summary['pooled_hidden_mean_mse_fraction'],rel_tol=1e-12)
    assert round(100*contrast/(common+contrast),1)==79.6
    assert all(float(r['light_esmda_within_block_mmd'])<float(r['light_prior_within_block_mmd']) for r in rows)
    print('5: layer errors, pooled 79.6% decomposition and 90 fine-detail MMD comparisons agree.')

    broken=[]
    for file in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
        for raw in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            if raw.startswith(('http:','https:','#','mailto:')): continue
            target=raw.split('#')[0]
            if target and not (file.parent/target).exists(): broken.append((str(file.relative_to(ROOT)),raw))
    assert not broken, broken
    print('Documentation: all local file links resolve.')


if __name__=='__main__': main()

"""Run an experiment with portable data and upstream import paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments'


def commands(task,data,skip_generation=False):
    if task=='1':
        code=E/'01_spatial_dependence'
        out=ROOT/'runs/01_spatial_dependence'
        return [
            [code/'poisson_reference_correlations.py','--case-source',data,'--output',out/'pixel'],
            [code/'poisson_patch_correlations.py','--case-source',data,'--output',out/'patch'],
            [code/'poisson_smooth_fields.py','--reference-curves',out/'patch/per_case.csv',
             '--training-metadata',data/'PDEFieldDataset_hf/Poisson/metadata.json','--output',out/'generator'],
            [code/'poisson_metric_detection.py','--design',out/'generator','--output',out/'detection','--workers','4'],
        ]
    if task=='2':
        return [[E/'02_sensor_placement/presentation_analysis.py','--output',ROOT/'runs/02_sensor_placement']]
    if task=='3':
        code=E/'03_posterior_geometry/esmda_paper_comparison'
        generation=[] if skip_generation else [[code/'run_comparison.py']]
        return generation+[[code/'analyze_dimensions.py'],[code/'dimensions/plot_pca_slide_maps.py']]
    if task=='4':
        code=E/'04_darcy_consistency'
        generation=[] if skip_generation else [[code/'darcy_esmda/reproduce.py']]
        return generation+[[code/'darcy_pressure'/name] for name in ('solve_pressures.py','analyze_pressures.py','plot_results.py')]
    code=E/'05_ltmi_recovery'
    generation=[] if skip_generation else [[code/'reproduce.py']]
    return generation+[[code/'posterior_analysis/analyze.py']]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=list('12345'))
    parser.add_argument('--data',type=Path,default=Path(os.environ.get('POSTERIORBENCH_DATA',str(ROOT/'data'))))
    parser.add_argument('--skip-generation',action='store_true',help='Analyse already-generated ensembles for studies 3-5')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    env=os.environ.copy()
    env['POSTERIORBENCH_DATA']=str(args.data.resolve())
    env['PYTHONPATH']=str(ROOT/'vendor/PosteriorBench')+os.pathsep+env.get('PYTHONPATH','')
    env['MPLBACKEND']='Agg'
    env['MPLCONFIGDIR']=str(ROOT/'runs/.matplotlib')
    env['PYTHONDONTWRITEBYTECODE']='1'
    for command in commands(args.experiment,args.data.resolve(),args.skip_generation):
        command=[sys.executable,*map(str,command)]
        print(subprocess.list2cmdline(command),flush=True)
        if not args.dry_run: subprocess.run(command,cwd=ROOT,env=env,check=True)

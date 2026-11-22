"""Download only the selected pinned PosteriorBench inputs, outside Git."""
import argparse
import os
from pathlib import Path

REVISION='f45719309a0ea6500b86e6e9973c18b3ffff8650'
ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['poisson','darcy','ltmi'])
    parser.add_argument('--output',type=Path,default=Path(os.environ.get('POSTERIORBENCH_DATA',str(ROOT/'data'))))
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    if args.task=='poisson':
        patterns=['PosteriorDataset_hf/Poisson/**','PDEFieldDataset_hf/Poisson/metadata.json',
                  'PDEFieldDataset_hf/Poisson/data-00000-of-00014.arrow']
    else:
        folder={'darcy':'Darcy','ltmi':'LTMI'}[args.task]
        patterns=[f'PosteriorDataset_hf/{folder}/**',f'PDEFieldDataset_hf/{folder}/**']
    snapshot_download(repo_id='jcy20/PosteriorBench',repo_type='dataset',revision=REVISION,
                      allow_patterns=patterns,local_dir=str(args.output))
    print(f'Pinned {args.task} inputs available at {args.output}')

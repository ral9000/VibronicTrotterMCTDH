"""
A script for submitting all jobs with slurm files in a directory at once.
"""

import os
import glob 
import subprocess

DIR = './EXPS_C60ANTH/dt=0.1'


slurm_files = glob.glob(os.path.join(DIR, '*.slurm'))

for file in slurm_files:
    print(f'Submitting {file}')
    #subprocess.run(['sbatch', file], check=True)
"""
A script for finding (& submitting) all slurm job files in a `runs` subdirectory at once.
"""

import os
import glob
import subprocess

DELTATs = [0.001, 0.01, 0.1, 1.0]

DO_RUN = False  # if True, attempts to submit the jobs to the slurm scheduler.

for DELTAT in DELTATs:

    DIR = f"./runs/EXPS_DABNA/dt={DELTAT}"
    print(DIR)
    os.chdir(DIR)
    slurm_files = glob.glob("*.slurm")

    for file in slurm_files:
        print(f"Found {file}")
        if DO_RUN:
            print(f"!! Submitting {file} !!")
            subprocess.run(["sbatch", file], check=True)
    os.chdir("../../..")

"""
Basic script for generating all required MCTDH directories and input files, and
automatically submitting the jobs to a slurm queue.
"""
import pickle 
import numpy as np 
from driver_mk_inputs import mk_input
from mode_selector import get_reduced_model
from mode_combiner import combine_by_frequency
from pennylane.labs.pf.vibronic import (VibronicMatrix,
                                     VibronicHamiltonian,
                                    )


"""
PARAMETERS
"""
NUM_MODES = 4
NUM_STATES = 4
DELTAT = 0.5
TMAX = 100
VIBHAMFILE = './VCHLIB/anthra-c60_ct_M=11.pkl'
DIRNAME = 'EXPS_C60ANTH'
RUNNAME = f'c60anth{NUM_STATES}s{NUM_MODES}m'
SUBMIT_SLURM = False
PTHREADS = 64

"""
Load a vibronic Hamiltonian
"""

filehandler = open(VIBHAMFILE, 'rb')
omegas_total, couplings = pickle.load(filehandler)
omegas_total = np.array(omegas_total)

"""
Parameterize simulation instances, and generate working directories and input files.
"""

lambdas = couplings[0]
alphas = couplings[1]
betas = couplings[2]
m_max = len(omegas_total)

assert NUM_MODES <= m_max

omegas, couplings_red = get_reduced_model(omegas_total, couplings, m_max=NUM_MODES, order_max=2, strategy=None)
lambdas = couplings_red[0]
alphas = couplings_red[1]
betas = couplings_red[2]

print(f'>>Constructing vibronic Hamiltonian for M = {NUM_MODES}...')
h_operator = VibronicHamiltonian(states = NUM_STATES, 
                                 modes = NUM_MODES, 
                                 phis = [lambdas, alphas, betas],
                                 omegas = omegas,
                                 sparse = True)

mode_labels = [f'mode{idx+1}' for idx in range(len(omegas))]
logical_modes = combine_by_frequency(mode_labels, omegas, target_binsize=4)

DIRNAME_M = f'c60anth4s{NUM_MODES}m'
PARENT_SUBDIR = f"./data/EXPS_C60ANTH/dt={DELTAT}"

mk_input(PARENT_SUBDIR,
        DIRNAME_M,
        h_operator, #VibronicHamiltonian 
        DELTAT,
        n_grid=8, 
        logical_modes = logical_modes,
        spfs_per_state = 6, 
        init_state=0, 
        t_max=TMAX, 
        pthreads=PTHREADS,
        exact=False, #overrides the spf specification with an exact calculation
        no_err_inps = False,
        make_slurm_file=True,
        )

"""
Submit the MCTDH jobs via slurm 
"""

if SUBMIT_SLURM:

    import os
    import glob 
    import subprocess

    slurm_files = glob.glob(os.path.join(PARENT_SUBDIR, '*.slurm'))

    for file in slurm_files:
        print(f'Submitting {file}')
        subprocess.run(['sbatch', file], check=True) 

print('**************\nMAIN_RUN.py successfully terminating.\n**************')
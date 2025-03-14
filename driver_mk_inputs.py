
"""
Driver to generate all required MCTDH files and directories.
"""

import os
from datetime import datetime
import numpy as np 
import pickle
from time import time 
from py_to_op import generate_op
from py_to_inp import generate_inp, generate_spf_scheme
from mode_selector import rank_modes, get_truncated_coupling_arrays, truncate_by_states
from pennylane.labs.vibronic import VibronicMatrix


def mk_input(parent_dir,
             name,
             h_operator, #VibronicHamiltonian 
             deltaT,
             n_grid, 
             logical_modes,
             spfs_per_state,  
             init_state, 
             t_max, 
             pthreads=10,
             exact=False, #overrides the spf specification with an exact calculation
             no_err_inps = False,
             make_slurm_file=False,
            ): 
    
    """
    parent_dir: directory 
    name: name
    h_operator: instance of VibronicHamiltonian
    deltaT: deltaT used in construction of effective Hamiltonian 
    n_grid: gridsize used for primitive basis 
    n_spfs_per_run: a list of integers. if you only want input files for 1 calculation, give a list of length 1.
        the integers in the list specify the number of SPFs per mode to use for that run. for instance, 
        n_spfs_per_run = [1,4] will generate input files for 2 separate calculations: one using 1 SPF per mode,
        i.e., a TDH calculation, and one using 4 SPFs per mode, a correlated calculation. 
    init_state: integer specifying which diabat should be initially populated to unity. 
    t_max: propagation time of the calculation(s) in femtoseconds 
    pthreads: number of threads MCTDH program will use for calculations 
    exact: if True, n_spfs_per_run is overridden and only a single exact calculation is performed.     
    """

    n = h_operator.states
    m = h_operator.modes

    print(type(h_operator))
    if isinstance(h_operator, VibronicMatrix):
        h_op = h_operator
    else:
        h_op  = h_operator.block_operator()

    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)  # Create the directory and any necessary parent directories
        print(f"\n>Parent deltaT directory created at: {parent_dir}")
    else:
        print(f"\n>Parent deltaT directory {parent_dir} already exists")

    str_op, metadata = generate_op(h_op, m, run_name=f'{name}')
    print(f'>>Writing vibronic Hamiltonian to {parent_dir}/{name}.op')
    file_handler = open(f'{parent_dir}/{name}.op', 'w')
    file_handler.write(str_op)
    file_handler.close()

    if not no_err_inps:
        print(f'>>Constructing effective Hamiltonian for M = {m} and deltaT = {deltaT}...')
        t0 = time()
        err_operator = h_operator.epsilon(delta=deltaT)
        h_eff_operator = h_op + err_operator
        print(f'      Obtained effective Hamiltonian after {round(time() - t0, 3)} s.')
        t0 = time()
        str_op, metadata = generate_op(h_eff_operator, m, run_name=f'{name}_err')
        print(f'      Obtained operator string for effective Hamiltonian after {round(time() - t0, 3)} s.')
        print(f'>>Writing effective vibronic Hamiltonian to {parent_dir}/{name}_err.op')
        file_handler = open(f'{parent_dir}/{name}_err.op', 'w')
        file_handler.write(str_op)
        file_handler.close()

    ##MODIFY THIS, no longer using spf in naming, or at all     
    name_extra = ''

    if exact: 
        spf_specs = 'exact'
        name_extra = 'exact'

    print(f'\n>>>Preparing directory and input file for standard Hamiltonian (exact = {exact})...')
    str_input = generate_inp(n_states=n, 
                            n_modes=m, 
                            n_grid=n_grid, 
                            logical_modes=logical_modes,
                            spfs_per_state = spfs_per_state,
                            init_state=init_state, 
                            t_max=t_max, 
                            run_name=f'{name}', 
                            opname=name, 
                            alloc_sec = metadata, 
                            pthreads=pthreads)

    #make directory if it doesnt exist
    directory = f"{parent_dir}/{name}"
    # Check if the directory exists
    if not os.path.isdir(directory):
        os.makedirs(directory)  # Create the directory and any necessary parent directories
        print(f">>>Directory created at: {directory}")
    else:
        print(">>>Directory already exists")

    file_handler = open(f'{parent_dir}/{name}.inp', 'w')
    file_handler.write(str_input)
    file_handler.close()

    if not no_err_inps:
        print(f'\n>>>Preparing directory and input file for effective Hamiltonian (exact = {exact})...') 
        str_input = generate_inp(n_states=n, 
                                n_modes=m, 
                                n_grid=n_grid, 
                                logical_modes=logical_modes,
                                spfs_per_state= spfs_per_state,
                                init_state=init_state, 
                                t_max=t_max, 
                                run_name=f'{name}_err', 
                                opname=f'{name}_err', 
                                alloc_sec = metadata, 
                                pthreads=pthreads)

        directory = f"{parent_dir}/{name}_err"
        # Check if the directory exists
        if not os.path.isdir(directory):
            os.makedirs(directory)  # Create the directory and any necessary parent directories
            print(f">>>Directory created at: {directory}")
        else:
            print(">>>Directory already exists")

        file_handler = open(f'{parent_dir}/{name}_err.inp', 'w')
        file_handler.write(str_input)
        file_handler.close()

    if make_slurm_file:
        generate_slurm_file(dir=parent_dir, job_name = f'{name}', cores=pthreads)
        if not no_err_inps:
            generate_slurm_file(dir=parent_dir, job_name = f'{name}_err', cores=pthreads)

def generate_slurm_file(dir, job_name, cores):

    slurm_str = f"""#!/bin/bash
# Slurm script automatically generated by mk_input at: {datetime.now()}
# Job name:
#SBATCH --job-name=mctdh_{job_name}
#
# Partition:
#SBATCH --partition=xanadu-internal
#
# Request one node:
#SBATCH --nodes=1
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case:
#SBATCH --cpus-per-task={cores}
#
# Ensure environment-modules are available in the executing job
source /etc/profile.d/modules.sh
# Command(s) to run:
mctdh86 -w {job_name}
"""
    file_handler = open(f'{dir}/{job_name}.slurm', 'w')
    file_handler.write(slurm_str)
    file_handler.close()

    print(f"Created slurm submit job script {job_name}.slurm at directory: {dir}")


if __name__ == "__main__":

    states = [1, 3]
    n = len(states)
    k = 8
    t_max = 1000
    init_state = 1 #python notation
    EXACT = True

    deltaT_series = [0.001, 0.01, 0.1, 1, 10]
    mode_series = [2,3,4]

    for m in mode_series:
        for deltaT in deltaT_series:

                name = f'no4a_{n}s{m}m'
                #Create a deltaT directory if doesnt exist:
                parent_dir = f"./{name}/deltaT={deltaT}"
                # Check if the directory exists
                if not os.path.isdir(parent_dir):
                    os.makedirs(parent_dir)  # Create the directory and any necessary parent directories
                    print(f"\n>Parent deltaT directory created at: {parent_dir}")
                else:
                    print(f"\n>Parent deltaT directory {parent_dir} already exists")

                print(f'\n>>Getting Vibronic Hamiltonian for n={n}, m={m}.')

                pkl_file = open('n4o4a_sf.pkl', 'rb')
                omegas, couplings = pickle.load(pkl_file)
                pkl_file.close()

                couplings_list = [couplings[c] for c in couplings]
                couplings_list = couplings_list[:3] #ignore cubic terms for now

                RANKING = 'C1N+F'

                couplings_list = truncate_by_states(couplings_list, states_to_keep = states)
                sorted_modes = rank_modes(omegas, couplings_list, ranking_type=RANKING)

                #Obtain truncated coupling_array 
                selected_modes = list(sorted_modes.keys())[:m]
                truncated_couplings = get_truncated_coupling_arrays(couplings_list, selected_modes, only_diagonal=False)
                truncated_omegas = [omegas[i] for i in selected_modes]
                omegas = np.array(truncated_omegas)
                lambdas, alphas, betas = truncated_couplings

                print(f'>>Constructing vibronic Hamiltonian for M = {m}...')
                h_operator = VibronicHamiltonian(n, m, alphas, betas, lambdas, omegas)


                str_op, metadata = generate_op(h_operator.block_operator(), m, run_name=f'{name}')
                
                # print(str_op)
                # # exit()

                print(f'>>Writing vibronic Hamiltonian to {parent_dir}/{name}.op')

                file_handler = open(f'{parent_dir}/{name}.op', 'w')
                file_handler.write(str_op)
                file_handler.close()

                print(f'>>Constructing effective Hamiltonian for M = {m} and deltaT = {deltaT}...')

                err_operator = h_operator.epsilon(delta=deltaT)
                h_eff_operator = h_operator.block_operator() + err_operator
                str_op, metadata = generate_op(h_eff_operator, m, run_name=f'{name}_err')

                print(f'>>Writing effective vibronic Hamiltonian to {parent_dir}/{name}_err.op')
                file_handler = open(f'{parent_dir}/{name}_err.op', 'w')
                file_handler.write(str_op)
                file_handler.close()

                spf_series = [8]
                for spf_num in spf_series:

                    print(f'\n>>>Preparing directory and input file for standard Hamiltonian (n_spf = {spf_num})...')
                    if not EXACT:  
                        n_spf_per_mode = [str(spf_num)]*n
                        n_spf_per_mode  = ','.join(n_spf_per_mode)
                        n_spfs = [n_spf_per_mode]*m
                        name_extra = f'spf{spf_num}'

                    else:
                        n_spfs = 'exact'
                        name_extra = 'exact'

                    str_input = generate_inp(n_states=n, 
                                            n_modes=m, 
                                            n_grid=k, 
                                            n_spfs = n_spfs, 
                                            init_state=init_state, 
                                            t_max=t_max, 
                                            run_name=f'{name}_{name_extra}', 
                                            opname=name, 
                                            alloc_sec = metadata, 
                                            pthreads=pthreads)

                    #make directory if it doesnt exist

                    directory = f"{parent_dir}/{name}_{name_extra}"
                    # Check if the directory exists
                    if not os.path.isdir(directory):
                        os.makedirs(directory)  # Create the directory and any necessary parent directories
                        print(f">>>Directory created at: {directory}")
                    else:
                        print(">>>Directory already exists")

                    file_handler = open(f'{parent_dir}/{name}_{name_extra}.inp', 'w')
                    file_handler.write(str_input)
                    file_handler.close()

                    print(f'\n>>>Preparing directory and input file for effective Hamiltonian (n_spf = {spf_num})...') 
                    str_input = generate_inp(n_states=n, 
                                            n_modes=m, 
                                            n_grid=k, 
                                            n_spfs = 
                                            n_spfs, 
                                            init_state=init_state, 
                                            t_max=t_max, 
                                            run_name=f'{name}_{name_extra}_err', 
                                            opname=f'{name}_err', 
                                            alloc_sec = metadata, 
                                            pthreads=pthreads)

                    directory = f"{parent_dir}/{name}_{name_extra}_err"
                    # Check if the directory exists
                    if not os.path.isdir(directory):
                        os.makedirs(directory)  # Create the directory and any necessary parent directories
                        print(f">>>Directory created at: {directory}")
                    else:
                        print(">>>Directory already exists")

                    file_handler = open(f'{parent_dir}/{name}_{name_extra}_err.inp', 'w')
                    file_handler.write(str_input)
                    file_handler.close()
                    
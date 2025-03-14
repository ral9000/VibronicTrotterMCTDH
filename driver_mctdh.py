
"""
Driver to automatically set up jobs needed for the MCTDH effective dynamics experiments.
"""
import os 
import time 


# specify m and deltaT from cmd line 

# then submit a mctdh job for each n_spf in spf_range specified 


def run_time(t1, t0, hrs=True):
    if hrs: 
        return f'{round((t1 - t0) / 3600, 4)} hrs'
    return f'{round((t1 - t0) / 3600, 4)} sec'

def execute(dir, name, no_err_calc = False):
    """
    Function for running the mctdh calculation
    """
    t0 = time.time()

    command = f'mctdh86 -w {dir}/{name}'
    print(f'>running {command}')
    t0_std = time.time()
    #run standard dynamics 
    os.system(command)
    t1_std = time.time()
    t_std = run_time(t1_std, t0_std)
    print(f'Finished MCTDH run of standard dynamics after {t_std}.')

    if not no_err_calc:
        t0_eff = time.time()
        #run effective dynamics 
        command = f'mctdh86 -w {dir}/{name}_err'
        print(f'>running {command}')
        
        os.system(command)
        
        t1_eff = time.time()
        t_eff = run_time(t1_eff, t0_eff)
        print(f'Finished MCTDH run of effective dynamics after {t_eff}.')

def gen_jobs(name, deltaT, m, spf_range):
    """
    Function for generating slurm job bash scripts 
    """
    return 1 

def submit_jobs(name, deltaT, m, spf_range):
    """
    Function for submitting the slurm jobs from the bash scripts
    """
    return 1


#deltaT_series = [0.016, 0.031, 0.062, 0.125, 0.25, 0.5]
#mode_series = [18]#[2,4,6,8,10,12,14,16]

if __name__ == "__main__":

    deltaT_series = [0.001, 0.01, 0.1, 1, 10]
    mode_series = [2,3,4]

    for m in mode_series:
        for deltaT in deltaT_series:
            dir = f'./no4a_2s{m}m/deltaT={deltaT}'
            name = f'no4a_2s{m}m_exact'
            spf_range = [8]
            execute(dir, name)
            

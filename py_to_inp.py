import numpy as np 



def generate_spf_scheme(logical_modes, spfs_per_state, n_states):

    spf_nums_str = ','.join([str(spfs_per_state) for _ in range(n_states)])
    scheme_str = ''
    for logical_mode in logical_modes: 
        logical_mode_str = ', '.join(logical_mode)
        scheme_str += f'{logical_mode_str} = {spf_nums_str}\n'

    return(scheme_str)


def generate_primitive_basis_section(n, m, n_grid, basis='HO'):

    input_str = ''
    for idx in range(m):
        input_str += f'    mode{idx+1}     {basis}     {n_grid}   0.0   1.0   1.0\n'
    input_str += f'    el        el     {n}'
    return input_str


def generate_init_wf_section(n, m, init_state = 0, basis='HO'):

    input_str = f'build\ninit_state = {init_state+1}'
    for idx in range(m):
        input_str += f'\nmode{idx+1}       {basis}      0.0    0.0    1.0'
    input_str += '\nend-build'
    return input_str

def generate_spf_basis_section(m, logical_modes, spfs_per_state, n_states, multi_set=True):
    
    """
    Currently only supports single set
    """

    if multi_set:
        input_str = 'multi-set\n'

    if len(logical_modes) == 1:
        spfs_per_state = 1 #avoids redundant configurations

    input_str += generate_spf_scheme(logical_modes, spfs_per_state, n_states)

    return input_str

def generate_inp(n_states, 
                 n_modes, 
                 n_grid, 
                 logical_modes,
                 spfs_per_state, 
                 init_state, 
                 t_max=1000, 
                 run_name=None, 
                 alloc_sec = None, 
                 opname=None, 
                 pthreads=1,
                 EXACT=False):
    exact_keyword = ''
    integrator_sec = """\nINTEGRATOR-SECTION
                          #nohsym
                          CMF/var = 0.5,  1.0d-05
                          BS/spf  =  7 ,  1.0d-05 ,  2.5d-04
                          SIL/A   =  5 ,  1.0d-05
                          #CMF
                          #RK5 = 1.0d-7
                          end-integrator-section
                          """
    if EXACT:
        exact_keyword = 'exact'
        integrator_sec  = """\nINTEGRATOR-SECTION
                               #nohsym
                               end-integrator-section
                          """
    else:
        spf_basis_section_str = generate_spf_basis_section(n_modes, logical_modes, spfs_per_state, n_states)
    primitive_basis_section_str = generate_primitive_basis_section(n_states, n_modes, n_grid, basis='HO')
    init_wf_section_str = generate_init_wf_section(n_states, n_modes,init_state = init_state, basis='HO')
    default_alloc_str = """maxmuld = 10000
maxfac = 10000
maxkoe = 10000
maxhtm = 10000
maxhop = 10000
"""
    if alloc_sec is None:
        alloc_str = default_alloc_str

    else:
        alloc_str = ''
        for key in alloc_sec:
            alloc_str += f'{key} = {alloc_sec[key]}\n'

    if opname is None:
        opname = run_name 

    file_str = f"""
RUN-SECTION
usepthreads = {pthreads}
name = {run_name}
title = {run_name}
propagation {exact_keyword}
tfinal = {t_max}  
tout = 5.0  
tpsi = 5.0
geninwf
psi
#genpes
auto
end-run-section
\nOPERATOR-SECTION
opname = {opname}
end-operator-section
\nALLOC-SECTION
{alloc_str}
end-alloc-section
\nSPF-BASIS-SECTION
{spf_basis_section_str}
end-spf-basis-section
\nPRIMITIVE-BASIS-SECTION
{primitive_basis_section_str}
end-primitive-basis-section
{integrator_sec}
\nINIT_WF-SECTION
{init_wf_section_str} 
end-init_wf-section
\nend-input
"""
    return file_str


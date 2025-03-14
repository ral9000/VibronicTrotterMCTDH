"""
PARAMETERS
"""
VIBHAMFILE = './VCHLIB/anthra-c60_ct_M=11.pkl'
DIRNAME = 'EXPS_C60ANTH'
RUNNAME = 'c60anth4s'
DIROUTPUT = 'DATA_C60ANTH'
n = 4
Ms = [4]
DELTATs = [0.1]

"""
Read the population results 
"""

for deltaT in DELTATs:
    for m in Ms:

        FILEOUTPUT = f'c60anth4s{m}m_dT={deltaT}'
        PARENT_SUBDIR = f"./EXPS_C60ANTH/dt={deltaT}"

        from driver_rdcheck import read_populations
        std_pops_mctdh, eff_pops_mctdh, errors_mctdh = read_populations(PARENT_SUBDIR, f'{RUNNAME}{m}m', n_states=n, no_err_read=False)

        """
        Save the population results 
        """

        import pickle 
        data = [std_pops_mctdh, eff_pops_mctdh, errors_mctdh]
        pickle.dump(data, DIROUTPUT)

print('**************\nMAIN_INTERPRET.py successfully terminating.\n**************')
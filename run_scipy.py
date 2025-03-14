import re
import numpy as np 
from vibronic import VibronicHamiltonian, coeffs
from sympy import var
import pickle 
import numpy as np
import matplotlib.pyplot as plt 

pkl_file = open('n4o4a_sf.pkl', 'rb')
omegas, couplings = pickle.load(pkl_file)
pkl_file.close()

from mode_selector import rank_modes, get_truncated_coupling_arrays

couplings_list = [couplings[c] for c in couplings]
couplings_list = couplings_list[:3] #ignore cubic terms for now

#Get vibronic coupling arrays
RANKING = 'C1N+F'
n_modes = 4
n_states = 4
sorted_modes = rank_modes(omegas, couplings_list, ranking_type=RANKING)
#Obtain truncated coupling_array 
selected_modes = list(sorted_modes.keys())[:n_modes]
truncated_couplings = get_truncated_coupling_arrays(couplings_list, selected_modes, only_diagonal=False)
truncated_omegas = [omegas[i] for i in selected_modes]
state_truncated_couplings = []
for coupling_arr in truncated_couplings:
    if len(np.shape(coupling_arr)) == 2:  #const
        state_truncated_couplings.append(coupling_arr[:n_states, :n_states])
    elif len(np.shape(coupling_arr)) == 3: #lin
        state_truncated_couplings.append(coupling_arr[:n_states, :n_states, :])
    elif len(np.shape(coupling_arr)) == 4: #quad
        state_truncated_couplings.append(coupling_arr[:n_states, :n_states, :, :])
truncated_couplings = state_truncated_couplings
omegas = np.array(truncated_omegas)

#Get vibronic Hamiltonian 

t = 1000
n = 4
m = 1

lambdas, alphas, betas = truncated_couplings

h_operator = VibronicHamiltonian(n, m, alphas, betas, lambdas, omegas)
err_operator = h_operator.epsilon(delta=1)
h_eff_operator = h_operator.block_operator() + err_operator

#TO DO:
#Use scipy to compute time series of observables for time evolution under the effective Hamiltonian 
#Compare with MCTDH exact results. 
#Attempt to run MCTDH for effective Hamiltonian for whole M=19 N4O40Anth model 
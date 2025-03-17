
# Description  

This code is for ...

# Usage 

## 1. Generating the MCTDH input files, and instantiating the data file

mention that this instantiates a data file with instance parameters (number of states, number of modes, couplings, etc.). The full list of parameters is:

- `N_states` : Number of electronic states, `int` 
- `M_modes` : Number of vibrational modes, int 
- `deltaT` : The Trotter timestep used in obtaining the Trotter effective Hamiltonian, `float`
- `Tmax` :  The maximum time propagation, `float` 
- `omegas` : The vibrational mode frequencies, `list`
- `couplings` The vibronic coupling arrays, `list[np.arrays]` 
- `path` : The path to the corresponding MCTDH run folder, `str` 
- `init_state` : The index of the initial electronic state used in the dynamics simulation, `int` 
- `exact` : Whether an exact MCTDH calculation was specified for this run, `bool`

## 2. Running the MCTDH calculations 

mention that this can be done by setting `DO_RUN` to True in the run scripts, or just use something like script `submit_jobs.py`  if you want to submit them later.

## 3. Extracting the population trajectories and errors, and updating the data file. 

getting the pop_trajectories 


If everything has succesfully completed, loading the datafile with pickle with, e.g.,

```python
with open("./data/no4a_5s_6m_t500_dt=1.0.pkl", "rb") as file:
    data = pickle.load(file)
```

should provide a dictionary `data` with the same keys as listed before, but with 3 extra keys:

- `std_trajectories` : a dictionary with keys being timestamps in femtoseconds, and values being the electronic state populations as propagated by the standard Hamiltonian.
- `eff_trajectories` : same as for `std_trajectories`, but propagated by the Trotter effective Hamiltonian.
- `error_trajectories` : a dictionary with keys being timestamps in femtoseconds, with values being the absolute magnitude of differences between electronic state populations propagated by the standard and Trotter effective Hamiltonian.

1) Pennylane branch `product-formula-framework`: https://github.com/PennyLaneAI/pennylane/tree/product-formula-framework

2) An installation of the Heidelberg MCTDH package version 8 release 6. 
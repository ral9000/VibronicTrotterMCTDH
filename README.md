
# Description  

This code is for using MCTDH vibronic dynamics simulations to estimate the effect of Trotter approximation on diabatic electronic state populations. 

# Usage 

To generate a complete datapoint with input parameters (i.e., specification of the model and Trotter timestep) and output parameters (i.e., the error trajectories), there are currently three steps that must be done. The first is specifying the parameters which go into the MCTDH simulations. The second is running the MCTDH simulations. Third and finally, reading out the time trajectories of the Trotter error from the MCTDH output files. 

## 1. Generating the MCTDH input files, and instantiating the data file

Examples:
- `MAIN_RUN_dabna.py`
- `MAIN_RUN_no4a.py` 

These files are used to instantiate a data file, containing a list of parameters describing the vibronic Hamiltonian and simulation setup. They also generate all the required MCTDH folders and files from which the simulations can be ran. All MCTDH working directories are contained in subdirectory `./runs`. The full list of parameters included in the data dictionary are:
- `N_states` : Number of electronic states, `int` 
- `M_modes` : Number of vibrational modes, `int` 
- `deltaT` : The Trotter timestep used in obtaining the Trotter effective Hamiltonian, `float`
- `Tmax` :  The maximum time propagation, `float` 
- `omegas` : The vibrational mode frequencies, `list`
- `couplings` The vibronic coupling arrays, `list[np.arrays]` 
- `path` : The path to the corresponding MCTDH run folder, `str` 
- `init_state` : The index of the initial electronic state used in the dynamics simulation, `int` 
- `exact` : Whether an exact MCTDH calculation was specified for this run, `bool`
This data dictionary is saved as a `pickle` object in directory `./data`.

## 2. Running the MCTDH simulations

This can be done by setting `DO_RUN=True `in the `MAIN_RUN_*` scripts, or just use script `submit_jobs.py`  if you want to submit them later. Setting `DO_RUN=True` or using `submit_jobs.py` work in a similar fashion: both will submit slurm jobs corresponding to any slurm files found in the directory. 

## 3. Extracting the population trajectories and errors, and updating the data file. 

Examples:
- `MAIN_INTERPRET_dabna.py`
- `MAIN_INTERPRET_no4a.py` 

After the MCTDH calculations have completed, we need to update the corresponding data files in `./data` to include the error trajectories we have obtained. This is done as shown in the `MAIN_INTERPRET_*` scripts. These scripts load the previously generated dictionary (generated when running `MAIN_RUN_*`), as well as the population trajectories stored in MCTDH output files, and update the data dictionary to include the population and Trotter error trajectories. If everything has succesfully completed, loading the datafile with pickle with, e.g.,
```python
with open("./data/no4a_5s_6m_t500_dt=1.0.pkl", "rb") as file:
    data = pickle.load(file)
```
should provide a dictionary `data` with the same keys as listed before, but with 3 extra keys:
- `std_trajectories` : a dictionary with keys being timestamps in femtoseconds, and values being the electronic state populations as propagated by the standard Hamiltonian.
- `eff_trajectories` : same as for `std_trajectories`, but propagated by the Trotter effective Hamiltonian.
- `error_trajectories` : a dictionary with keys being timestamps in femtoseconds, with values being the absolute magnitude of differences between electronic state populations propagated by the standard and Trotter effective Hamiltonian.

If these dictionaries are empty, its likely that the MCTDH simulations have not been ran yet. 

# Dependencies 

1) Pennylane branch `product-formula-framework`: https://github.com/PennyLaneAI/pennylane/tree/product-formula-framework

2) An installation of the Heidelberg MCTDH package version 8 release 6. 
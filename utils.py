import math
from time import time
import numpy as np
from scipy.sparse import diags
from scipy.linalg import dft # For the Discrete Fourier Transform matrix
from scipy.sparse import csr_matrix
from scipy.sparse import kron, eye, block_array
from scipy.sparse.linalg import svds, matrix_power, expm, norm, expm_multiply
import itertools
from functools import lru_cache
from scipy.linalg import eigh


def fs_to_inverse_ev(t_fs):

    #Units of Hamiltonian:
    energy_unit = 'eV'

    #Desired evolution time in femtoseconds:
    t_desired_fs = t_fs
    t_desired_s = t_desired_fs *  1e-15

    #Atomic unit to SI conversions:
    hbar = 1.054571817 * 1e-34 #J*s

    J_to_eV = 6.2415 * 1e18 #J/eV = 6.2415 * 1e18

    E_J = hbar / t_desired_s 
    E_eV = E_J * J_to_eV
    t_eV = 1 / E_eV

    return t_eV

def get_pows2(n):
    binary_repr = bin(n)[:1:-1]
    return [2 ** i for i, bit in enumerate(binary_repr) if bit == '1']

def pad_to_power_of_two(arr):
    n = arr.shape[0]

    dim = len(arr.shape)
    if dim > 2: 
        m = np.shape(arr)[-1]

    n_pad = 2 ** math.ceil(math.log2(n))  # Smallest power of 2 >= n

    # Create a new (m, m) array filled with zeros

    # Place the original (n, n) array in the upper-left corner
    if dim == 2: 
        padded_array = np.zeros((n_pad, n_pad), dtype=arr.dtype)
        padded_array[:n, :n] = arr

    elif dim == 3: 
        padded_array = np.zeros((n_pad, n_pad, m), dtype=arr.dtype)
        padded_array[:n, :n, :] = arr

    elif dim == 4: 
        padded_array = np.zeros((n_pad, n_pad, m, m), dtype=arr.dtype)
        padded_array[:n, :n, :, :] = arr
    return padded_array


def get_propagator(hamiltonian):
    def propagator(t):
        return expm(1j * t * hamiltonian)
    return propagator

def compute_eiht(t, propagator, memo):
    
    """
    eiht : sparse matrix representing e^iHt with t = 1.
    t : total propagation time t 
    sub_evolutions 
    """
    powers = get_pows2(t)
    dim = np.shape(memo[1])[0]
    result = eye(dim)
    for pow in powers:
        if pow not in memo:
            memo[pow] = propagator(pow)
        result @= memo[pow]
    return result

def commutator(A, B):
    # Calculate the commutator of sparse matrices A and B
    return A @ B - B @ A

def nested_commutator(matrices, pattern):
    # Initialize the commutator to the first matrix in the pattern
    com = matrices[pattern[-1]]
    for i in range(len(pattern)-2, -1, -1):
        com = commutator(matrices[pattern[i]], com)
    return com

def nested_kron(matrices):
    # Efficiently compute the Kronecker product of a list of sparse matrices
    result = matrices[0]
    for matrix in matrices[1:]:
        result = kron(result, matrix, format='csr')
    return result


def spectral_norm(matrix):
    # Compute the largest singular value for a sparse matrix
    # u, s, vt = svds(matrix, k=1, return_singular_vectors=False)  # Adjust k if more stability is needed

    s = svds(matrix, k=1, return_singular_vectors=False)
    return s[0]

def trace_2norm(matrix):
    #Compute the trace (Schatten 2-norm) for a sparse matrix. 
    M = matrix @ np.conjugate(matrix).T
    return M.trace()

def position_operator(dimension, power):
    values = ((np.arange(dimension) - dimension / 2) * (np.sqrt(2 * np.pi / dimension))) ** power
    return diags(values, 0, format='csr')

def momentum_operator(dimension, power):
    values = np.arange(dimension)
    values[dimension // 2:] -= dimension
    values = (values * (np.sqrt(2 * np.pi / dimension))) ** power
    DFT = dft(dimension, scale='sqrtn')  # Get DFT matrix scaled by sqrt(N)
    matrix = DFT @ np.diag(values) @ DFT.conj().T

    return csr_matrix(matrix)  # Convert to sparse format after computation

def modify_bits(i: int, j: int, n: int) -> int:
    # Convert i and j to n-bit binary strings
    i_bin = f'{i:0{n}b}'
    j_bin = f'{j:0{n}b}'
    # Convert binary strings to lists for easier manipulation
    i_list = list(i_bin)

    # Apply NOT operation where j has a 1
    for idx in range(n):
        if j_bin[idx] == '1':
            i_list[idx] = '1' if i_list[idx] == '0' else '0'

    # Convert the modified list back to a binary string
    modified_i_bin = ''.join(i_list)
    # Convert the binary string back to an integer
    return int(modified_i_bin, 2)

def effective_subspace_projector(dimension, threshold=1e-9):
    vals, vecs = eigh((1j*commutator(position_operator(dimension, 1), momentum_operator(dimension, 1))).todense())
    indices = np.where(np.abs(vals - 1) <= threshold)[0]
    basis = [vecs[:, i] for i in indices]
    return csr_matrix(sum(np.outer(basis[i], basis[i].conj()) for i in range(len(basis))))

def trotterize(fragments, total_time, n_steps):
    dim = np.shape(fragments[0])[0]
    result = eye(dim)
    for fragment in fragments:
        result = expm(1j * fragment * total_time / (n_steps*2)) @ result
    for fragment in fragments[::-1]:
        result = expm(1j * fragment * total_time / (n_steps*2)) @ result
    return matrix_power(result, n_steps)


def trace_norm(M, P=None):
    d = np.shape(M)[0]
    if P is None: 
        P = eye(d)
    tr = (M.getH() @ M @ P).diagonal().sum()

    d_imageP = P.diagonal().sum()
    return np.sqrt(tr) / np.sqrt(d_imageP)  

def find_num_steps(actual, fragments, total_time, epsilon, initial_guess = 10, norm_type='spectral', projector=None):
    """
    norm_type (str): must either be "spectral" or "trace". 

    projector (sparse matrix): if supplied, replaces norm calculations of U_actual - U_estimated by (U_actual - U_estimated)*P where P is 
        the projector supplied. 
    
    """

    if norm_type not in ['spectral', 'trace']:
        raise ValueError(f'Norm type "{norm_type}" not recognized.')

    if projector is None: #set to identity
        dim = np.shape(actual)[0] 
        projector = eye(dim)

    # Initial coarse search to find an upper bound
    n_steps = initial_guess
    while True:
        estimated = trotterize(fragments, total_time, n_steps)
        if norm_type == 'spectral':
            error = spectral_norm((actual - estimated) @ projector)
        elif norm_type == 'trace':
            error = trace_norm(actual - estimated, P = projector)
        if error <= epsilon:
            break
        n_steps *= 2
    
    if abs(error) <= 1e-10: #consider this exact. no need for further search.
        return n_steps 
    # Perform binary search between n_steps // 2 and n_steps
    low = n_steps // 2
    high = n_steps
    while low < high:
        mid = (low + high) // 2
        estimated = trotterize(fragments, total_time, mid)
        if norm_type == 'spectral':
            error = spectral_norm((actual - estimated) @ projector)
        elif norm_type == 'trace':
            error = trace_norm(actual - estimated, P = projector)
        if error <= epsilon:
            high = mid  # Narrow the upper bound
        else:
            low = mid + 1  # Narrow the lower bound

    return low

def find_num_steps_state_dependent(actual, fragments, total_time, epsilon, state, initial_guess = 10):
    """
    Finds the r value which satisfies eps < |U_exact|psi> - U_approx|psi>|
    """

    # Initial coarse search to find an upper bound
    n_steps = initial_guess
    while True:
        estimated = trotterize(fragments, total_time, n_steps)
        error = state_specific_error(U_exact=actual, U_approx=estimated, psi=state)
        if error <= epsilon:
            break
        n_steps *= 2
    
    if abs(error) <= 1e-10: #consider this exact. no need for further search.
        return n_steps 
    # Perform binary search between n_steps // 2 and n_steps
    low = n_steps // 2
    high = n_steps
    while low < high:
        mid = (low + high) // 2
        estimated = trotterize(fragments, total_time, mid)
        error = state_specific_error(U_exact=actual, U_approx=estimated, psi=state)
        if error <= epsilon:
            high = mid  # Narrow the upper bound
        else:
            low = mid + 1  # Narrow the lower bound

    return low

def find_num_steps_expectation_dependent(actual, fragments, total_time, epsilon, state, observables, initial_guess = 10):
    """
    Finds the r value which satisfies eps < |<psi|U_exact O U_exact|psi> - <psi|U_approx O U_approx|psi>| 
    """
    
    # Initial coarse search to find an upper bound
    n_steps = initial_guess
    while True:
        estimated = trotterize(fragments, total_time, n_steps)
        errors = [expectation_value_error(U_exact=actual, U_approx=estimated, psi=state, O=observable) for observable in observables] 
        if max(errors) <= epsilon:
            break
        n_steps *= 2
    
    if abs(max(errors)) <= 1e-10: #consider this exact. no need for further search.
        return n_steps 
    # Perform binary search between n_steps // 2 and n_steps
    low = n_steps // 2
    high = n_steps
    while low < high:
        mid = (low + high) // 2
        estimated = trotterize(fragments, total_time, mid)
        errors = [expectation_value_error(U_exact=actual, U_approx=estimated, psi=state, O=observable) for observable in observables] 
        if max(errors) <= epsilon:
            high = mid  # Narrow the upper bound
        else:
            low = mid + 1  # Narrow the lower bound
    return low

def find_num_steps_com(fragments, total_time, eps):
    result = 0
    for j, fragment in enumerate(fragments):
        for k, fragment2 in enumerate(fragments[j+1:]):
            fragment3 = fragment + 2 * np.sum(fragments[j+1:])
            result += commutator(fragment3, commutator(fragment, fragment2))

    W = spectral_norm(result)/24
    return int(total_time**1.5 *(np.sqrt(W/eps)))

def error_matrix(fragments):
    

    n = fragments[0].shape[0]
    error_op = csr_matrix((n,n))
    for j, fragment in enumerate(fragments):
        for k, fragment2 in enumerate(fragments[j+1:]):
            fragment3 = fragment + 2 * np.sum(fragments[j+1:])
            error_op += commutator(fragment3, commutator(fragment, fragment2))

    return error_op / 24

def hogs_statevec(n_modes, n_grid):
    """
    State vector for the tensor product of harmonic oscillator ground states. 
    """
    values = (np.arange(n_grid) - n_grid / 2) * (np.sqrt(2 * np.pi / n_grid))
    ideal = np.exp(-values**2 / 2)
    ideal /= np.linalg.norm(ideal)

    # if n_modes == 1:
    #     return ideal.reshape(1, n_grid)
    if n_modes == 1:
        return ideal.reshape(n_grid, 1)

    result = ideal
    for _ in range(1, n_modes):
        result = kron(result, ideal)
    return result.T


def vertically_excited_state(i, n_states, n_modes, n_grid):
    """
    Returns the vibronic wavefunction that is the i^th electronic basis state tensor product with the product of harmonic oscillator ground states. 
    """
    hogs = hogs_statevec(n_modes, n_grid)
    wavepackets = [np.zeros((n_grid**n_modes, 1)) for _ in range(n_states)]

    if not isinstance(hogs, np.ndarray):
        wavepackets[i] = hogs.toarray()
    else:
        wavepackets[i] = hogs
    #concatenate the wavepackets
    wf = wavepackets[0]
    for wp in wavepackets[1:]:
        wf = np.concatenate((wf, wp), axis=0)
    return wf

def compute_observable_spectral_norm_error(U_exact, U_approx, observable, projector = None):
    obs_error_op = U_exact.getH() @ observable @ U_exact - U_approx.getH() @ observable @ U_approx 
    if projector is not None:
        obs_error_op = projector @ obs_error_op @ projector 
    return spectral_norm(obs_error_op)

def find_num_steps_observable_spectral(exact_propagator, h_fragments, total_time, epsilon, observables, initial_guess = 10, projector=None):

    # Initial coarse search to find an upper bound
    n_steps = initial_guess
    while True:
        estimated = trotterize(h_fragments, total_time, n_steps)
        errors = [compute_observable_spectral_norm_error(exact_propagator, estimated, O, projector) for O in observables] 
        if max(errors) <= epsilon:
            break
        n_steps *= 2
    
    if abs(max(errors)) <= 1e-10: #consider this exact. no need for further search.
        return n_steps 
    # Perform binary search between n_steps // 2 and n_steps
    low = n_steps // 2
    high = n_steps
    while low < high:
        mid = (low + high) // 2
        estimated = trotterize(h_fragments, total_time, mid)
        errors = [compute_observable_spectral_norm_error(exact_propagator, estimated, O, projector) for O in observables] 
        if max(errors) <= epsilon:
            high = mid  # Narrow the upper bound
        else:
            low = mid + 1  # Narrow the lower bound

    return low
    
def compute_rvals(h_fragments, total_time, eps, init_guess = 10):
    """
    returns the r values from exact trotter error spectral norm and that estimated via commutators. 
    """
    r_commutator = find_num_steps_com(h_fragments, total_time, eps)
    exact_propagator = expm(sum(h_fragments) * 1j * total_time)
    r_exact = find_num_steps(exact_propagator, h_fragments, total_time, eps, init_guess)
    return r_exact, r_commutator

def r_trotter_steps_from_exact_spectral(total_time, epsilon, exact_propagator, h_fragments, init_guess):
    t0 = time()
    r = find_num_steps(exact_propagator, h_fragments, total_time, epsilon, init_guess, norm_type = 'spectral')
    print(f'Found r = {r} for EXACT_SPECTRAL w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_spectral_projected(total_time, epsilon, exact_propagator, h_fragments, projector, init_guess):
    t0 = time()
    r = find_num_steps(exact_propagator, h_fragments, total_time, epsilon, init_guess, norm_type = 'spectral', projector = projector)
    print(f'Found r = {r} for EXACT_SPECTRAL_PROJECTED w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_commutator_spectral(total_time, epsilon, h_fragments):
    t0 = time()
    r = find_num_steps_com(h_fragments, total_time, epsilon)
    print(f'Found r = {r} for EXACT_COMMUTATOR_SPECTRAL w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_trace_norm(total_time, epsilon, exact_propagator, h_fragments, init_guess):
    t0 = time()
    r = find_num_steps(exact_propagator, h_fragments, total_time, epsilon, init_guess, norm_type = 'trace')
    print(f'Found r = {r} for EXACT_TRACE w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_trace_norm_projected(total_time, epsilon, exact_propagator, h_fragments, projector, init_guess):
    t0 = time()
    r = find_num_steps(exact_propagator, h_fragments, total_time, epsilon, init_guess, norm_type = 'trace', projector = projector)
    print(f'Found r = {r} for EXACT_TRACE_PROJECTED w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_state_norm(total_time, epsilon, exact_propagator, h_fragments, state, init_guess):
    t0 = time()
    r = find_num_steps_state_dependent(exact_propagator, h_fragments, total_time, epsilon, state, initial_guess = init_guess)
    print(f'Found r = {r} for EXACT_STATE w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r 

def r_trotter_steps_from_exact_expectationvalue(total_time, epsilon, exact_propagator, h_fragments, state, observables, init_guess):
    t0 = time()
    r = find_num_steps_expectation_dependent(exact_propagator, h_fragments, total_time, epsilon, state, observables, initial_guess = init_guess)
    print(f'Found r = {r} for EXACT_EXPECTATION w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_observable_spectral(total_time, epsilon, exact_propagator, h_fragments, observables, init_guess):
    """
    || U_trot' O U_trot - U' O U   || 
    """
    t0 = time()
    r = find_num_steps_observable_spectral(exact_propagator, h_fragments, total_time, epsilon, observables, initial_guess = init_guess, projector=None)
    print(f'Found r = {r} for EXACT_OBSERVABLE_SPECTRAL w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r

def r_trotter_steps_from_exact_projected_observable_spectral(total_time, epsilon, exact_propagator, h_fragments, observables, projector, init_guess):
    """
    ||P U_trot' O U_trot P - P U' O U P||
    """
    t0 = time()
    r = find_num_steps_observable_spectral(exact_propagator, h_fragments, total_time, epsilon, observables, initial_guess= init_guess, projector=projector)
    print(f'Found r = {r} for EXACT_OBSERVABLE_SPECTRAL_PROJECTED w/ t={total_time}, error < {epsilon} after {round(time() - t0, 2)} seconds ')
    return r 

def compute_observable_error(observable, exact_propagator, trotter_propagator, projector=None, norm_type='spectral'):
    """
    Given: `exact_propagator` e^iHt
           `trotter_propagator` U_trot
           `observable` O 
           optional `projector` P, defaults to identity
           and `norm type` || . ||, defaults to spectral 

    Returns:

        || e^(-iHt) O e^(iHt0) - U_trot' O U_trot ||
    """

    dim = np.shape(exact_propagator)[0]
    if projector is None: 
        projector = eye(dim)
    
    evolved_O_exact = exact_propagator.getH() @ observable @ exact_propagator 
    evolved_O_trotter = trotter_propagator.getH() @ observable @ trotter_propagator 

    uou_diff = evolved_O_exact - evolved_O_trotter 

    
    if norm_type == 'spectral':
        return spectral_norm(uou_diff)
    
    elif norm_type == 'trace':
        return trace_norm(uou_diff)

def linear_fit(x_data, y_data):

    # Calculate linear fit for commutator-based r 
    coefficients = np.polyfit(x_data, y_data, 1)  # Degree 1 for linear fit
    x = np.array(x_data)
    y= np.array(y_data)
    m, c = coefficients  # Slope and intercept

    # Generate fitted y-values
    y_fit = m * x + c

    # Calculate R^2
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_fit) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return m,c, r_squared

def state_specific_error(U_exact, U_approx, psi):

    """
    error = | U_exact|psi> - U_approx|psi> |
    """

    err_vec = U_exact @ psi - U_approx @ psi 
    return np.linalg.norm(err_vec)


def expectation_value(U, psi, O):
    
    return psi.conj().T @ U.conj().T @ O @ U @ psi 

def expectation_value_error(U_exact, U_approx, psi, O, return_pops = False):
    """
    error specific to state |psi> for operator O 

    error = | <psi|U_exact O U_exact|psi> - <psi|U_approx O U_approx|psi> |

    """

    o_exact = psi.conj().T @ U_exact.conj().T @ O @ U_exact @ psi 
    o_exact = o_exact[0][0]
    o_approx = psi.conj().T @ U_approx.conj().T @ O @ U_approx @ psi 
    o_approx = o_approx[0][0]
    err = abs(o_exact - o_approx)

    if return_pops:
        return err, o_exact, o_approx

    return err

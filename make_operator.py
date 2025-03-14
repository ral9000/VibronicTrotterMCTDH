import math
import numpy as np 
from copy import deepcopy
from itertools import product
from utils import position_operator, momentum_operator, modify_bits, spectral_norm, hogs_statevec
from scipy.sparse import kron, eye, csr_matrix, lil_matrix, block_array, block_diag

def build_operator_basis(n_modes, n_grid):
    #Constructs the operator for a linear combination of non-interacting harmonic oscillators
    id = eye(n_grid)
    Qs = []
    for i in range(n_modes):
        factors = [id] * n_modes
        factors[i] = position_operator(n_grid, power=1)
        tensor_product = factors[0]
        for factor in factors[1:]:
            tensor_product = kron(tensor_product, factor)
        Qs.append(tensor_product)

    Ps = []
    for i in range(n_modes):
        factors = [id] * n_modes
        factors[i] = momentum_operator(n_grid, power=1)
        tensor_product = factors[0]
        for factor in factors[1:]:
            tensor_product = kron(tensor_product, factor)
        Ps.append(tensor_product)

    return Qs, Ps 

def make_h0(omegas, operator_basis):
    "makes a harmonic oscillator"
    Qs, Ps = operator_basis    
    T = sum([omegas[i] / 2 * p @ p for i, p in enumerate(Ps)])
    V = sum([omegas[i] / 2 * q @ q for i, q in enumerate(Qs)])
    return T, V 

def coupling_array_to_sparse_polynomial(coupling_array, position_operators):

    """
    given a coupling array, constructs the homogeneous polynomial over position operators {Q}. coupling array specifies the coefficients
    entering this polynomial. the degree of the polynomial is inferred by the shape of `coupling_array`

    coupling_array: array of couplings. must be of the shape (M,), (M,M), (M,M,M), etc.

    position_operators: list of sparse matrices specifying the position operators for Q_0 to Q_(M-1).
    """
    deg = len(np.shape(coupling_array))
    n_modes = len(position_operators)
    dim = np.shape(position_operators[0])[0]
    if deg == 0:
        return coupling_array * eye(dim)
    M = np.shape(coupling_array)[0]
    if M != len(position_operators):
        raise ValueError(f'Expected M={M} from the shape of `coupling_array`, instead got {len(position_operators)} position operators!')
        
    ranges = [range(M)] * deg
    # Compute the Cartesian product of these ranges
    idxs = list(product(*ranges))
    #for each tuple of indices inside idxs, construct the product of the position operators from those indices: e.g.:
    #idx = idxs[0] = [0,1,2]. we want: coupling_array[idx] * Q[0] * Q[1] * Q[2]. 
    polynom = csr_matrix(np.zeros((dim,dim)))
    for idx_tup in idxs:
        #print(f'term: {idx_tup}')
        monomial = eye(dim)
        for idx in idx_tup:
            monomial = position_operators[idx] @ monomial 
        polynom += coupling_array[idx_tup] * monomial 

    return polynom


def make_vibronic_hamiltonian_matrix_elements(omegas, couplings, n_grid):

    """
    - `omegas` must be a list of floats. The number of modes `M` in the model is determined as `M = len(omegas)`.

    - `couplings` must be a list of arrays. The first array in `couplings` must be shape `(N,N)`, and
    specifies the zeroth order couplings. Number of states `N` is determined from np.shape(couplings[0])[0].
    The next array in couplings determines the linear couplings, and must have shape `(N, N, M)`. The next array
    determines the 2nd order couplings and must have shape `(N,N,M,M)`. 3rd order: `(N,N,M,M,M)`, and so on.

    - `n_grid` determines the size of the underlying grid for each mode. 
    """
    
    n_modes = len(omegas)
    Qs, Ps = build_operator_basis(n_modes, n_grid)
    dim = np.shape(Qs[0])[0]
    T, V = make_h0(omegas, [Qs, Ps]) #need to add V to all diagonal matrix elements! 

    if couplings is not None:
        n_states = np.shape(couplings[0])[0]
    else:
        n_states =  1
    #construct matrix elements 
    matrix_elements = {(i,j): None for i in range(n_states) for j in range(n_states) if i <= j}
    for i,j in matrix_elements:
        print((i,j))
        #print(f'matrix element (i={i}, j={j})')
        total_polynom_ij = csr_matrix(np.zeros((dim,dim)))
        couplings_ij = [couplings[o][i,j] for o in range(len(couplings))]
        print(couplings_ij)
        
        for c in couplings_ij: #iterates over the orders of coupling arrays for matrix element ij
            print(f'coupling: {c}')
            total_polynom_ij += coupling_array_to_sparse_polynomial(c, Qs)        
        print(f'total_polynom_{i}{j}')
        print(total_polynom_ij)
        matrix_elements[(i,j)] = total_polynom_ij
        if i == j:
            matrix_elements[(i,j)] = matrix_elements[(i,j)] + V
    return matrix_elements, T

def hermitize(matrix_elements):

    for i,j in list(matrix_elements.keys()):
        if i != j:
            matrix_elements[(j,i)] = matrix_elements[(i,j)]
    return matrix_elements

def get_vibronic_hamiltonian_fragments(omegas, couplings, n_grid):

    n_states = np.shape(couplings[0])[0]
    matrix_elements, kinetic = make_vibronic_hamiltonian_matrix_elements(omegas, couplings, n_grid)
    matrix_elements = hermitize(matrix_elements)
    #using `modify_bits`
    fragments_list = []
    for m in range(n_states):
        frag_matrix_elements = {}
        for j in range(n_states):
            jxorm = modify_bits(m, j, n_states)
            frag_matrix_elements[(j, jxorm)] = matrix_elements[(j, jxorm)]
        fragments_list.append(frag_matrix_elements)
    kinetic_fragment = {}
    for m in range(n_states):
        kinetic_fragment[(m,m)] = kinetic 
    fragments_list.append(kinetic_fragment)

    return fragments_list

def make_vibronic_hamiltonian_fragments_sparse(hamiltonian_fragments_list):
    # Initialize an empty block layout
    N = len(hamiltonian_fragments_list)
    # Fill the block array
    fragments_sparse = []
    for fragment in hamiltonian_fragments_list: 
        blocks = [[None for _ in range(N)] for _ in range(N)]
        for (i, j), matrix in fragment.items():
            blocks[i][j] = matrix
            print(blocks)
        fragments_sparse.append(block_array(blocks))
    return fragments_sparse

def make_vibgs_projector(n_modes, n_states, n_grid):
    hogs = hogs_statevec(n_modes, n_grid)
    proj_vib = hogs @ hogs.conj().T
    #pad in electronic matrix 
    blocks = [[None for _ in range(n_states)] for _ in range(n_states)]
    for i in range(n_states):
        blocks[i][i] = proj_vib
    return block_array(blocks)


def make_population_observable(i, n_states, n_modes, n_grid):

    "i: index of state to create observable |i><i| for."

    if i > n_states - 1: 
        raise ValueError(f'index i = {i} not supported for n_states = {n_states}.')

    vib_identity = eye(n_grid ** n_modes)
    blocks = [vib_identity if idx == i else csr_matrix((n_grid ** n_modes, n_grid ** n_modes)) for idx in range(n_states)]    #pad in electronic matrix 
    return block_diag(blocks, format='csr')

#pop_i_obs = make_population_observable(i=3, n_states = 2, n_modes = 2, n_grid = 2

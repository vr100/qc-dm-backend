import itertools
import numpy as np
from qiskit.quantum_info import DensityMatrix

def compare(aakash_result, qasm_result):
    aakash_dm = aakash_result["data"]["densitymatrix"]
    aakash_dm = np.asarray(np.round(aakash_dm, 4))

    qasm_sv = qasm_result.data.statevector
    n_qubits = qasm_result.header.n_qubits
    qasm_sv = swap_values_with_reverse(qasm_sv, n_qubits)
    qasm_dm = DensityMatrix(qasm_sv).data
    qasm_dm = np.asarray(np.round(qasm_dm, 4))

    return np.allclose(aakash_dm.real, qasm_dm.real) and \
        np.allclose(aakash_dm.imag, qasm_dm.imag)

## TODO: figure out why we need to swap
def swap_values_with_reverse(state_vector, n_qubits):
    rev_pairs = get_reverse_pairs(n_qubits)
    new_sv = state_vector.copy()
    for a,b in rev_pairs:
        if a >= b:
            continue
        # note: if a > b, already swapped
        # if a == b, no need to swap
        # so swap only for a < b
        new_sv[a], new_sv[b] = new_sv[b], new_sv[a]
    return new_sv

## pair up value with its reverse bits
## for example, let n_qubits = 3
## 000, 001, 010, 011, 100, 101, 110, 111
## pairs would be
## (0,0), (1,4), (2,2), (3,6),
## (4,1), (5,5), (6,3), (7,7)
def get_reverse_pairs(n_qubits):
    val_range = 2 ** n_qubits
    pairs = []
    for val in range(val_range):
        bin_str_val = "{:0{n}b}".format(val, n=n_qubits)
        reverse_str_val = bin_str_val[::-1]
        reverse_val = int(reverse_str_val, 2)
        pairs.append([val, reverse_val])
    return pairs

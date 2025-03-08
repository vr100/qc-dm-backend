import itertools
import numpy as np
from qiskit.quantum_info import DensityMatrix

def run_and_compare_without_measure(circ, backend_1, backend_2):
    job = backend_1.run(circ)
    actual_result = job.result()["results"][0]
    job = backend_2.run(circ)
    expected_result = job.result().results[0]
    return compare_dm(actual_result, expected_result)

def compare_dm(actual_result, expected_result):
    actual_dm = actual_result["data"]["densitymatrix"]
    actual_dm = np.asarray(np.round(actual_dm, 4))

    expected_sv = expected_result.data.statevector
    n_qubits = expected_result.header.n_qubits
    # reversing the bit ordering in state vector to
    # match the ordering of the actual result
    expected_sv = swap_values_with_reverse(expected_sv, n_qubits)
    expected_dm = DensityMatrix(expected_sv).data
    expected_dm = np.asarray(np.round(expected_dm, 4))

    return np.allclose(actual_dm.real, expected_dm.real) and \
        np.allclose(actual_dm.imag, expected_dm.imag)

def run_and_compare_with_measure(circ, measure_circ,
    measure_bits, backend_1, backend_2, shots=100):
    job = backend_1.run(measure_circ, shots=shots)
    actual_result = job.result()
    actual_prob = (list)(actual_result["results"][0]["data"]["partial_probability"].values())

    job = backend_2.run(circ)
    expected_result = job.result()
    expected_sv = expected_result.results[0].data.statevector
    expected_prob = np.array(expected_sv.probabilities(qargs=measure_bits))
    return np.allclose(actual_prob, expected_prob)

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
        new_sv._data[a], new_sv._data[b] = state_vector._data[b], state_vector._data[a]
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

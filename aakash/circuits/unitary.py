import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aakash import AAKASH_DM
from utils import run_and_compare_without_measure, \
    run_and_compare_with_measure
from qiskit_aer import StatevectorSimulator
from qiskit.circuit.library import CCXGate, CCZGate, CSwapGate, iSwapGate

def ciswapgate():
    zero = np.array([1, 0]).reshape((2,1))
    one = np.array([0, 1]).reshape((2,1))
    i = np.eye(2)
    iswap = iSwapGate().to_matrix()
    ciswapgate = np.kron(zero @ zero.T, np.kron(i, i)) + \
        np.kron(one @ one.T, iswap)
    return ciswapgate

def test_unitary_gates():
    backend1 = AAKASH_DM.get_backend('dm_simulator',
        filters=lambda x: x.name == "dm_simulator")
    backend2 = StatevectorSimulator()
    backend2.SHOW_FINAL_STATE = True
    options = {}

    q = QuantumRegister(4, 'q')
    c = ClassicalRegister(3, 'c')
    circ = QuantumCircuit(q, c)

    ccx = CCXGate().to_matrix()
    ccz = CCZGate().to_matrix()
    cswap = CSwapGate().to_matrix()
    ciswap = ciswapgate()

    circ.h(0)
    circ.x(2)
    circ.h(3)
    circ.cx(2,3)
    circ.unitary(ccx, [0,1,2])
    circ.unitary(ccz, [1,2,3])
    circ.x(1)
    circ.h(3)
    circ.unitary(ciswap, [0,2,3])
    circ.unitary(cswap, [1,2,0])
    circ.tdg(1)

    before_success = run_and_compare_without_measure(circ,
        backend1, backend2)

    measure_circ = circ.copy()
    measure_circ.measure(q[0], c[0])
    measure_circ.measure(q[2], c[1])
    measure_circ.measure(q[3], c[2])

    after_success = run_and_compare_with_measure(circ, measure_circ,
        [3, 2, 0], backend1, backend2)

    print(f"Comparing results before measurement: {before_success}")
    print(f"Comparing results after measurement: {after_success}")

def main():
  test_unitary_gates()

if __name__ == "__main__":
  main()

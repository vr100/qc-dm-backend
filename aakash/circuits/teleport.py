import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aakash import AAKASH_DM
from utils import run_and_compare_without_measure, \
    run_and_compare_with_measure
from qiskit_aer import StatevectorSimulator

backend1 = AAKASH_DM.get_backend('dm_simulator',
    filters=lambda x: x.name == "dm_simulator")
backend2 = StatevectorSimulator()
backend2.SHOW_FINAL_STATE = True
options = {}

q = QuantumRegister(3, 'q')
c = ClassicalRegister(3, 'c')
circ = QuantumCircuit(q, c)

circ.h(q[1])
circ.cx(q[1], q[2])
circ.h(q[0])
circ.cx(q[0], q[1])
circ.h(q[0])

before_success = run_and_compare_without_measure(circ,
    backend1, backend2)

measure_circ = circ.copy()
measure_circ.measure(q[0], c[0])
measure_circ.measure(q[1], c[1])
if c[0] == 1:
    measure_circ.z(q[2])
if c[1] == 1:
    measure_circ.x(q[2])
measure_circ.measure(q[2], c[2])

after_success = run_and_compare_with_measure(circ, measure_circ,
    [2, 1, 0], backend1, backend2)

print(f"Comparing results before measurement: {before_success}")
print(f"Comparing results after measurement: {after_success}")

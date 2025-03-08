import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aakash import AAKASH_DM
from utils import run_and_compare_without_measure, run_and_compare_with_measure
from qiskit_aer import StatevectorSimulator

backend1 = AAKASH_DM.get_backend('dm_simulator',
	filters=lambda x: x.name == "dm_simulator")
backend2 = StatevectorSimulator()
backend2.SHOW_FINAL_STATE = True
options = {}

q = QuantumRegister(3, 'q')
c = ClassicalRegister(2, 'c')
circ = QuantumCircuit(q, c)

circ.x(q[0])
circ.h(q[1])
circ.h(q[2])
circ.h(q[0])
circ.h(q[0])
circ.cx(q[1], q[0])
circ.tdg(q[0])
circ.cx(q[2], q[0])
circ.t(q[0])
circ.cx(q[1], q[0])
circ.tdg(q[0])
circ.cx(q[2], q[0])
circ.t(q[0])
circ.tdg(q[1])
circ.h(q[0])
circ.cx(q[2], q[1])
circ.tdg(q[1])
circ.cx(q[2], q[1])
circ.s(q[1])
circ.t(q[2])
circ.h(q[1])
circ.h(q[2])
circ.x(q[1])
circ.x(q[2])
circ.h(q[1])
circ.cx(q[2], q[1])
circ.h(q[1])
circ.x(q[1])
circ.x(q[2])
circ.h(q[1])
circ.h(q[2])

before_success = run_and_compare_without_measure(circ, backend1, backend2)

measure_circ = circ.copy()
measure_circ.measure(q[1], c[0])
measure_circ.measure(q[2], c[1])

after_success = run_and_compare_with_measure(circ,
	measure_circ, backend1, backend2)

print()
print(f"Before measure comparison result: {before_success}")
print(f"After measure comparing results: {after_success}")
print()


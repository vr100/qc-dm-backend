import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.providers.aakash import AAKASH_DM
from utils import compare

backend1 = AAKASH_DM.get_backend('dm_simulator')
backend2 = AAKASH_DM.get_backend('qasm_simulator')
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
circ.measure(q[1], c[0])
circ.measure(q[2], c[1])


circuits = [circ]
job = execute(circuits, backend1, **options)
aakash_result = job.result()
print(aakash_result)

job = execute(circuits, backend2, **options)
qasm_result = job.result()
print(qasm_result)

success = compare(aakash_result["results"][0],
	qasm_result.results[0])
print(f"Comparing results: {success}")

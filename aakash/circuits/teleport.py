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
c = ClassicalRegister(3, 'c')
circ = QuantumCircuit(q, c)

circ.h(q[1])
circ.cx(q[1], q[2])
circ.h(q[0])
circ.cx(q[0], q[1])
circ.h(q[0])

print()
print("## Before measurement ##")
print()

circuits = [circ]
job = execute(circuits, backend1, **options)
aakash_result_before = job.result()
print(aakash_result_before)

job = execute(circuits, backend2, **options)
qasm_result_before = job.result()
print(qasm_result_before)

circ.measure(q[0], c[0])
circ.measure(q[1], c[1])
if c[0] == 1:
    circ.z(q[2])
if c[1] == 1:
    circ.x(q[2])
circ.measure(q[2], c[2])

print()
print("## After measurement ##")
print()

circuits = [circ]
job = execute(circuits, backend1, **options)
aakash_result = job.result()
print(aakash_result)

job = execute(circuits, backend2, **options)
qasm_result = job.result()
print(qasm_result)

print()
success = compare(aakash_result_before["results"][0],
    qasm_result_before.results[0])
print(f"Comparing results before measurement: {success}")

success = compare(aakash_result["results"][0],
    qasm_result.results[0])
print(f"Comparing results after measurement: {success}")

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.providers.aakash import AAKASH_DM
from utils import compare

backend1 = AAKASH_DM.get_backend('dm_simulator')
backend2 = AAKASH_DM.get_backend('qasm_simulator')
backend2.SHOW_FINAL_STATE = True
options = {}

def generator(k):
    return (np.pi*2)/(2**k)


num_of_qubits = 5
q = QuantumRegister(num_of_qubits, 'q')
circ = QuantumCircuit(q)

circ.h(q[0])
circ.cu(0.0, 0.0, generator(2), 0.0, q[1],q[0])

circ.h(q[1])
circ.cu(0.0, 0.0, generator(3), 0.0, q[2], q[0])
circ.cu(0.0, 0.0, generator(2), 0.0, q[2], q[1])

circ.h(q[2])
circ.cu(0.0, 0.0, generator(4), 0.0, q[3], q[0])
circ.cu(0.0, 0.0, generator(3), 0.0, q[3], q[1])
circ.cu(0.0, 0.0, generator(2), 0.0, q[3], q[2])

circ.h(q[3])
circ.cu(0.0, 0.0, generator(5), 0.0, q[4], q[0])
circ.cu(0.0, 0.0, generator(4), 0.0, q[4], q[1])
circ.cu(0.0, 0.0, generator(3), 0.0, q[4], q[2])
circ.cu(0.0, 0.0, generator(2), 0.0, q[4], q[3])

circ.h(q[4])

for wire in range (num_of_qubits-1):
    circ.h(q[wire])
    for rotation in range(wire+1):
        circ.cu(0.0, 0.0, generator(wire+2-rotation), 0.0, q[wire+1], q[rotation])
circ.h(q[num_of_qubits-1])

#circ.draw(output='mpl', line_length=120, scale=0.5)
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

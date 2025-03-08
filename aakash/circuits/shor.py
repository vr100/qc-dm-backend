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

# generate angle for controlled phase gate R(k)
def generator(k):
    return (np.pi*2)/(2**k)


num_of_qubits = 5
q = QuantumRegister(num_of_qubits, 'q')
c = ClassicalRegister(3,'c')
circ = QuantumCircuit(q,c)
circ.h(q[0])
circ.h(q[1])
circ.h(q[2])
circ.cx(q[2],q[3])
circ.cx(q[2],q[4])
circ.h(q[1])
circ.cu1(generator(2), q[1],q[0])
circ.h(q[0])
circ.cu1(generator(3), q[1], q[2])
circ.cu1(generator(2), q[0], q[2])
circ.h(q[2])


before_measure = run_and_compare_without_measure(circ,
    backend1, backend2)

measure_circ = circ.copy()
measure_circ.measure(q[0],c[0])
measure_circ.measure(q[1],c[1])
measure_circ.measure(q[2],c[2])

after_measure = run_and_compare_with_measure(circ, measure_circ,
    [2,1,0], backend1, backend2)

print(f"Comparing results before measurement: {before_measure}")
print(f"Comparing results after measurement: {after_measure}")

from qiskit.circuit.library import CU1Gate, U1Gate
from qiskit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit import InstructionSet
from qiskit.circuit import quantumcircuit
from qiskit.circuit.quantumcircuit import QubitSpecifier
from typing import Optional, Union

def cu1(
    self,
    theta: ParameterValueType,
    control_qubit: QubitSpecifier,
    target_qubit: QubitSpecifier,
    label: Optional[str] = None,
    ctrl_state: Optional[Union[str, int]] = None,
) -> InstructionSet:
    r"""Apply :class:`~qiskit.circuit.library.CU1Gate`.

    For the full matrix form of this gate, see the underlying gate documentation.

    Args:
        theta: The :math:`\lambda` rotation angle of the gate.
        control_qubit: The qubit(s) used as the control.
        target_qubit: The qubit(s) targeted by the gate.
        label: The string label of the gate in the circuit.
        ctrl_state:
            The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
            on the '1' state.

    Returns:
        A handle to the instructions created.
    """
    from qiskit.circuit.library import CU1Gate

    return self.append(
        CU1Gate(theta, label=label, ctrl_state=ctrl_state),
        [control_qubit, target_qubit],
        [],
    )

def u1(
    self,
    theta: ParameterValueType,
    qubit: QubitSpecifier,
) -> InstructionSet:
    r"""Apply :class:`~qiskit.circuit.library.UGate`.

    For the full matrix form of this gate, see the underlying gate documentation.

    Args:
        theta: The :math:`\lambda` rotation angle of the gate.
        qubit: The qubit(s) to apply the gate to.

    Returns:
        A handle to the instructions created.
    """
    from qiskit.circuit.library import U1Gate

    return self.append(U1Gate(theta), [qubit], [])

QuantumCircuit.cu1 = cu1
QuantumCircuit.u1 = u1


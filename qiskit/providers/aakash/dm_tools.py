# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains functions used by the basic aer simulators.

"""

from string import ascii_uppercase, ascii_lowercase
import numpy as np
from copy import deepcopy
from qiskit.exceptions import QiskitError
import itertools
from qiskit.providers.basic_provider.basic_provider_tools \
  import SINGLE_QUBIT_GATES, single_gate_matrix, \
  TWO_QUBIT_GATES, TWO_QUBIT_GATES_WITH_PARAMETERS
from .decompose_tools import get_gates_for_unitary

SINGLE_QUBIT_GATES_LIST = (list)(SINGLE_QUBIT_GATES.keys())
TWO_QUBIT_GATES_LIST =  (list)(TWO_QUBIT_GATES_WITH_PARAMETERS.keys()) + \
    (list)(TWO_QUBIT_GATES.keys())

def decompose_gates(unitary):
    gates = get_gates_for_unitary(unitary)
    compatible_gates_list = []
    for g in gates:
        param = None if len(g["params"]) == 0 else g["params"][0]
        compatible_gates_list.append([g["oper"], param])
    return compatible_gates_list

def single_gate_dm_matrix(gate, params=None):
    """Get the rotation matrix for a single qubit in density matrix formalism.

    Args:
        gate(str): the single qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        array: Decomposition in terms of 'ry', 'rz' with their angles. 
    """

    if gate in SINGLE_QUBIT_GATES_LIST:
        gate_matrix = single_gate_matrix(gate, params)
        decomp_gates = decompose_gates(gate_matrix)
        return decomp_gates
    else:
        raise QiskitError('Gate is not among the valid types: %s' % gate)

def decompose_gates_multiple(unitary):
    gates = get_gates_for_unitary(unitary)
    compatible_gates_list = []
    for g in gates:
        param = None if len(g["params"]) == 0 else g["params"][0]
        compatible_gates_list.append([g["oper"], param, *g["bits"]])
    return compatible_gates_list

def multi_gate_dm_matrix(gate, params=None):
    """Get the rotation matrix for two qubits in density matrix formalism.

    Args:
        gate(str): the two qubit gate name
        params(list): the operation parameters op['params']
    Returns:
        array: Decomposition in terms of 'ry', 'rz', 'cx', 'global phase'
    """

    if gate in TWO_QUBIT_GATES.keys():
        gate_matrix = TWO_QUBIT_GATES[gate]
        decomp_gates = decompose_gates_multiple(gate_matrix)
        return decomp_gates
    elif gate in TWO_QUBIT_GATES_WITH_PARAMETERS.keys():
        gate_matrix = TWO_QUBIT_GATES_WITH_PARAMETERS[gate](*params).to_matrix()
        decomp_gates = decompose_gates_multiple(gate_matrix)
        return decomp_gates
    elif gate == "unitary":
        gate_matrix = params[0]
        decomp_gates = decompose_gates_multiple(gate_matrix)
        return decomp_gates
    else:
        raise QiskitError('Gate is not among the valid types: %s' % gate)

def rot_gate_dm_matrix(gate, param, err_param, state, q, num_qubits):
    """   
    The error model adds a fluctuation to the angle param,
    with mean err_param[1] and variance parametrized in terms of err_param[0].
    
    Args:
        gate (string): Rotation axis
        param (float): Rotation angle
        err_param[1] is the mean error in the angle param.
        err_param[0] is the reduction in the radius after averaging over fluctuations in the angle param.
        state is the reshaped density matrix according to the gate location.
    """

    c = err_param[0] * np.cos(param + err_param[1])
    s = err_param[0] * np.sin(param + err_param[1])

    if gate == 'rz':
        k = [1, 2]
    elif gate == 'ry':
        k = [3, 1]
    elif gate == 'rx':
        k = [2, 3]
    else:
        raise QiskitError(
            'Gate is not among the valid decomposition types: %s' % gate)

    state1 = state.copy()
    temp1 = state1[:,k[0],:]
    temp2 = state1[:,k[1],:]

    state[:, k[0], :] = c * temp1 - s * temp2
    state[:, k[1], :] = c * temp2 + s * temp1

    return state


def U3_merge(xi, theta1, theta2):
    """ Performs merge operation when both the gates are U3,
        by transforming the Y-Z-Y decomposition of the gates to the Z-Y-Z decomposition.
        Args:
            [xi, theta1, theta2] (list, type:float ):  {Ry(theta1) , Rz(xi) , Ry(theta2)}
            0 <= theta1, theta2 <= Pi , 0 <= xi <= 2*Pi
        Return
            [β, α, γ] (list, type:float ):  {Rz(α) , Ry(β) , Rz(γ)}
            0 <= β <= Pi , 0 <= α, γ <= 2*Pi     

    Input Matrix Form
    {
        E^(-((I xi)/2))*cos[theta1/2]*cos[theta2/2] - 
        E^((I xi)/2)*sin[theta1/2]*sin[theta2/2]	(1,1)

       -E^(((I xi)/2))*sin[theta1/2]*cos[theta2/2] - 
        E^(-((I xi)/2))*cos[theta1/2]*sin[theta2/2]  (1,2)

        E^(-((I xi)/2))*sin[theta1/2]*cos[theta2/2] + 
        E^((I xi)/2)*cos[theta1/2]*sin[theta2/2]	(2,1)

        E^((I xi)/2)*cos[theta1/2]*cos[theta2/2] - 
        E^(-((I xi)/2))*sin[theta1/2]*sin[theta2/2]  (2,2)
    }
    Output Matrix Form
    {
        E^(-I(α + γ)/2)*cos[β/2]    -E^(-I(α - γ)/2)*sin[β/2]

        E^(I(α - γ)/2)*sin[β/2]     E^(I(α + γ)/2)*cos[β/2]
    }

    """

    sxi = np.sin(xi*0.5)
    cxi = np.cos(xi*0.5)
    sth1p2 = np.sin((theta1+theta2)*0.5)
    cth1p2 = np.cos((theta1+theta2)*0.5)
    sth1m2 = np.sin((theta1-theta2)*0.5)
    cth1m2 = np.cos((theta1-theta2)*0.5)

    apg2 = np.arctan2(sxi*cth1m2, cxi*cth1p2)
    amg2 = np.arctan2(-sxi*sth1m2, cxi*sth1p2)

    alpha = apg2 + amg2
    gamma = apg2 - amg2

    cb2 = np.sqrt((cxi*cth1p2)**2 + (sxi*cth1m2)**2)
    beta = 2 * np.arccos(cb2)

    return beta, alpha, gamma


def mergeU(gate1, gate2):
    """
    Merges Unitary Gates acting consecutively on the same qubit within a partition.
    Args:
        Gate1   ([Inst, index])
        Gate2   ([Inst, index])
    Return:
        Gate    ([Inst, index])
    """
    #print("Merged ",gate1[0].name, "qubit", gate1[0].qubits, " with ", gate2[0].name, "qubit", gate2[0].qubits)
    temp = None
    # To preserve the sequencing we choose the smaller index while merging.
    if gate1[1] < gate2[1]:
        temp = deepcopy(gate1)
    else:
        temp = deepcopy(gate2)

    if gate1[0].name == 'u1' and gate2[0].name == 'u1':
        temp[0].params[0] = gate1[0].params[0] + gate2[0].params[0]
    elif gate1[0].name == 'u1' or gate2[0].name == 'u1':
        # If first gate is U1
        if temp[0].name == 'u1':
            temp[0].name = 'u3'
            for i in range(2):
                temp[0].params.append(0)

        if gate1[0].name == 'u1' and gate2[0].name == 'u3':
            temp[0].params[0] = gate2[0].params[0]
            temp[0].params[1] = gate2[0].params[1]
            temp[0].params[2] = gate2[0].params[2] + gate1[0].params[0]
        elif gate1[0].name == 'u3' and gate2[0].name == 'u1':
            temp[0].params[0] = gate1[0].params[0]
            temp[0].params[1] = gate1[0].params[1] + gate2[0].params[0]
            temp[0].params[2] = gate1[0].params[2]
    elif gate1[0].name == 'u3' and gate2[0].name == 'u3':
        theta = float(gate2[0].params[2] + gate1[0].params[1]) 
        phi = float(gate2[0].params[0]) 
        lamb = float(gate1[0].params[0])

        res = U3_merge(theta, phi, lamb)

        temp[0].params[0] = res[0]
        temp[0].params[1] = gate2[0].params[1] + res[1]
        temp[0].params[2] = gate1[0].params[2] + res[2]
    else:
        raise QiskitError(
            'Encountered unrecognized instructions: %s, %s' % gate1[0].name, gate2[0].name)
    return temp


def merge_gates(inst):
    """
    Unitary rotation gates on a single qubit are merged iteratively,
    by combining consecutive gate pairs.
    Args:
        Inst [[inst, index]]:   Instruction list to be merged
    Return
        Inst [Qasm Inst]:       Merged instruction
    """

    if len(inst) < 2:
        return inst[0][0]
    else:
        temp = mergeU(inst[0], inst[1])
        for idx in range(2, len(inst)):
            param = []
            temp = mergeU(temp, inst[idx])
        return temp[0]


def single_gate_merge(inst, num_qubits, merge_flag=True):
    """
        Merges single gates applied consecutively to each qubit in the circuit.
        Args:
            inst [QASM Inst]:   List of instructions (original)
        Return
            inst [QASM Inst]:   List of instructions after merging
    """

    single_gt = [[] for x in range(num_qubits)]
    inst_merged = []

    if merge_flag:
        for ind, op in enumerate(inst):
            # To preserve the sequencing of the instructions
            opx = [op, ind]
            # Gates that are not single qubit rotations separate merging segments
            if opx[0].name in ('CX', 'cx', 'measure', 'bfunc', 'reset', 'barrier'):
                for idx, sg in enumerate(single_gt):
                    if sg:
                        inst_merged.append(merge_gates(sg))
                        single_gt[idx] = []
                if opx[0].name == 'CX':
                    opx[0].name = 'cx'
                inst_merged.append(opx[0])
            # Single qubit rotations are appended to their respective qubit instructions
            elif opx[0].name in ('U', 'u1', 'u2', 'u3'):
                if opx[0].name == 'U':
                    opx[0].name = 'u3'
                elif opx[0].name == 'u2':
                    opx[0].name = 'u3'
                    opx[0].params.insert(0, np.pi/2)
                single_gt[op.qubits[0]].append(opx)
            elif opx[0].name in ['id', 'u0']:
                continue
            elif opx[0].name in SINGLE_QUBIT_GATES_LIST:
                inst_merged.append(opx[0])
            elif opx[0].name in TWO_QUBIT_GATES_LIST:
                inst_merged.append(opx[0])
            elif opx[0].name == "unitary":
                inst_merged.append(opx[0])
            else:
                raise QiskitError('Encountered unrecognized instruction: %s' % op)

        # To merge the final remaining gates
        for gts in single_gt:
            if gts:
                inst_merged.append(merge_gates(gts))
    else:
        for op in inst:
            # Only names are changed without merging
            if op.name == 'CX':
                op.name = 'cx'
            elif op.name == 'U':
                    op.name = 'u3'
            elif op.name == 'u2':
                op.name = 'u3'
                op.params.insert(0, np.pi/2)
            
            if op.name not in ['id', 'u0']:
                inst_merged.append(op)

    return inst_merged


def cx_gate_dm_matrix(state, q_1, q_2, err_param, num_qubits):
    """Apply C-NOT gate in density matrix formalism.

        Args:
        state : density matrix
        q_1 (int): Control qubit 
        q_2 (int): Target qubit
        Note : Ordering of qubits (MSB right, LSB left)

    The error model adds a fluctuation "a" to the angle producing the X rotation,
    with mean err_param[1] and variance parametrized in terms of err_param[0].
    The noisy C-NOT gate then becomes (1 0 0 0), (0 1 0 0), (0 0 Isin(a) cos(a)), (0 0 cos(a) Isin(a))
    Args:
        err_param[1] is the mean error in the angle param "a".
        err_param[0] is the reduction in the radius after averaging over fluctuations in the angle param,
                     which equals <cos(a)>.
    """

    # Calculating all cos and sin in advance
    cav = err_param[0]
    c2av = 4*cav - 3 # assuming small fluctuations in angle "a"
    c = cav * np.cos(err_param[1])
    s = cav * np.sin(err_param[1])
    c2 = 0.5 * (1 + c2av * np.cos(2*err_param[1]))
    s2 = 0.5 * (1 - c2av * np.cos(2*err_param[1]))
    s = cav * np.sin(err_param[1])
    cs = c2av * np.sin(err_param[1]) * np.cos(err_param[1])

    if (q_1 == q_2) or (q_1 >= num_qubits) or (q_2 >= num_qubits):
        raise QiskitError('Qubit Labels out of bound in CX Gate')
    elif q_2 > q_1:
        # Reshape Density Matrix
        rt, mt2, ct, mt1, lt = 4**(num_qubits-q_2 - 1), 4, 4**(q_2-q_1-1), 4, 4**(q_1)
        state = np.reshape(state, (lt, mt1, ct, mt2, rt))
        temp_dm = state.copy()

        state[:, 0, :, 2, :] = s2*temp_dm[:, 0, :, 2, :] + c2*temp_dm[:, 3, :, 2, :] - \
                          cs*(temp_dm[:, 0, :, 3, :] - temp_dm[:, 3, :, 3, :])
        state[:, 3, :, 2, :] = c2*temp_dm[:, 0, :, 2, :] + s2*temp_dm[:, 3, :, 2, :] + \
                          cs*(temp_dm[:, 0, :, 3, :] - temp_dm[:, 3, :, 3, :])
        state[:, 0, :, 3, :] = s2*temp_dm[:, 0, :, 3, :] + c2*temp_dm[:, 3, :, 3, :] + \
                          cs*(temp_dm[:, 0, :, 2, :] - temp_dm[:, 3, :, 2, :]) 
        state[:, 3, :, 3, :] = c2*temp_dm[:, 0, :, 3, :] + s2*temp_dm[:, 3, :, 3, :] - \
                          cs*(temp_dm[:, 0, :, 2, :] - temp_dm[:, 3, :, 2, :])

        state[:, 1, :, 0, :] = c*temp_dm[:, 1, :, 1, :] - s*temp_dm[:, 2, :, 0, :]
        state[:, 1, :, 1, :] = c*temp_dm[:, 1, :, 0, :] - s*temp_dm[:, 2, :, 1, :]
        state[:, 1, :, 2, :] = -s*temp_dm[:, 2, :, 2, :] + c*temp_dm[:, 2, :, 3, :]
        state[:, 1, :, 3, :] = -c*temp_dm[:, 2, :, 2, :] - s*temp_dm[:, 2, :, 3, :]

        state[:, 2, :, 0, :] = s*temp_dm[:, 1, :, 0, :] + c*temp_dm[:, 2, :, 1, :]
        state[:, 2, :, 1, :] = s*temp_dm[:, 1, :, 1, :] + c*temp_dm[:, 2, :, 0, :]
        state[:, 2, :, 2, :] = s*temp_dm[:, 1, :, 2, :] - c*temp_dm[:, 1, :, 3, :]
        state[:, 2, :, 3, :] = c*temp_dm[:, 1, :, 2, :] + s*temp_dm[:, 1, :, 3, :]

    else:
        # Reshape Density Matrix
        rt, mt2, ct, mt1, lt = 4**(num_qubits-q_1 -1), 4, 4**(q_1-q_2-1), 4, 4**(q_2)
        state = np.reshape(state, (lt, mt1, ct, mt2, rt))
        temp_dm = state.copy()

        state[:, 2, :, 0, :] = s2*temp_dm[:, 2, :, 0, :] + c2*temp_dm[:, 2, :, 3, :] - \
                          cs*(temp_dm[:, 3, :, 0, :] - temp_dm[:, 3, :, 3, :])
        state[:, 2, :, 3, :] = c2*temp_dm[:, 2, :, 0, :] + s2*temp_dm[:, 2, :, 3, :] + \
                          cs*(temp_dm[:, 3, :, 0, :] - temp_dm[:, 3, :, 3, :])
        state[:, 3, :, 0, :] = s2*temp_dm[:, 3, :, 0, :] + c2*temp_dm[:, 3, :, 3, :] + \
                          cs*(temp_dm[:, 2, :, 0, :] - temp_dm[:, 2, :, 3, :]) 
        state[:, 3, :, 3, :] = c2*temp_dm[:, 3, :, 0, :] + s2*temp_dm[:, 3, :, 3, :] - \
                          cs*(temp_dm[:, 2, :, 0, :] - temp_dm[:, 2, :, 3, :])

        state[:, 0, :, 1, :] = c*temp_dm[:, 1, :, 1, :] - s*temp_dm[:, 0, :, 2, :]
        state[:, 1, :, 1, :] = c*temp_dm[:, 0, :, 1, :] - s*temp_dm[:, 1, :, 2, :]
        state[:, 2, :, 1, :] = -s*temp_dm[:, 2, :, 2, :] + c*temp_dm[:, 3, :, 2, :]
        state[:, 3, :, 1, :] = -c*temp_dm[:, 2, :, 2, :] - s*temp_dm[:, 3, :, 2, :]

        state[:, 0, :, 2, :] = s*temp_dm[:, 0, :, 1, :] + c*temp_dm[:, 1, :, 2, :]
        state[:, 1, :, 2, :] = s*temp_dm[:, 1, :, 1, :] + c*temp_dm[:, 0, :, 2, :]
        state[:, 2, :, 2, :] = s*temp_dm[:, 2, :, 1, :] - c*temp_dm[:, 3, :, 1, :]
        state[:, 3, :, 2, :] = c*temp_dm[:, 2, :, 1, :] + s*temp_dm[:, 3, :, 1, :]

    state =  np.reshape(state, num_qubits * [4])

    return state

def is_unitary(gate):
    # Checks if gate is unitary
    return True if gate.name == "unitary" else False

def is_single(gate):
    # Checks if gate is single
    return True if gate.name in SINGLE_QUBIT_GATES_LIST else False

def is_double(gate):
    # Checks if gate is CX
    return True if gate.name in TWO_QUBIT_GATES_LIST else False

def is_measure(gate):
    # Checks if gate is measure
    return True if gate.name == 'measure' else False

def is_reset(gate):
    # Checks if gate is reset
    return True if gate.name == 'reset' else False

def is_measure_dummy(gate):
    # Checks if gate is dummy measure
    return True if gate.name == 'dummy_measure' else False

def is_reset_dummy(gate):
    # Checks if gate is dummy reset
    return True if gate.name == 'dummy_reset' else False

def qubit_stack(i_set, num_qubits):
    """ Divides the sequential instructions for the whole register
        in to a stack of sequential instructions for each qubit.
        Multi-qubit instructions appear in the list for each involved qubit.
    Args:
        i_set (list): instruction set for the register
        num_qubits (int): number of qubits
    """

    instruction_set = [[] for _ in range(num_qubits)]
    for idx, instruction in enumerate(i_set):
        if not is_measure(instruction) and not is_reset(instruction):
            # instuctions are appended unless measure and reset
            for qubit in instruction.qubits:
                instruction_set[qubit].append(instruction)

        elif is_measure(instruction):
            if instruction_set[instruction.qubits[0]]:
                if not is_measure_dummy(instruction_set[instruction.qubits[0]][-1]):
                    instruction_set[instruction.qubits[0]].append(instruction)
                    dummy = deepcopy(instruction)
                    dummy.name = 'dummy_measure'
                    dummy.qubits[0] = -1
                    for qubit in set(range(num_qubits)).difference(set(instruction.qubits)):
                        instruction_set[qubit].append(dummy)
                else:
                    instruction_set[instruction.qubits[0]][-1] = instruction
            else:
                instruction_set[instruction.qubits[0]].append(instruction)
                dummy = deepcopy(instruction)
                dummy.name = 'dummy_measure'
                dummy.qubits[0] = -1
                for qubit in set(range(num_qubits)).difference(set(instruction.qubits)):
                    instruction_set[qubit].append(dummy)
        
        elif is_reset(instruction):
            if instruction_set[instruction.qubits[0]]:
                if not is_reset_dummy(instruction_set[instruction.qubits[0]][-1]):
                    instruction_set[instruction.qubits[0]].append(instruction)
                    dummy = deepcopy(instruction)
                    dummy.name = 'dummy_reset'
                    dummy.qubits[0] = -1
                    for qubit in set(range(num_qubits)).difference(set(instruction.qubits)):
                        instruction_set[qubit].append(dummy)
                else:
                    instruction_set[instruction.qubits[0]][-1] = instruction
            else:
                instruction_set[instruction.qubits[0]].append(instruction)
                dummy = deepcopy(instruction)
                dummy.name = 'dummy_reset'
                dummy.qubits[0] = -1
                for qubit in set(range(num_qubits)).difference(set(instruction.qubits)):
                    instruction_set[qubit].append(dummy)

    stack_depth = max([len(stack) for stack in instruction_set])
    return instruction_set, stack_depth


def partition_helper(i_set, num_qubits):
    """ Partitions the stack of qubit instructions in to a set of sequential levels.
        Instructions in a single level do not overlap and can be executed in parallel.
    """

    i_stack, depth = qubit_stack(i_set, num_qubits)
    level, sequence = 0, [[] for _ in range(depth)]

    while i_set:
        # Qubits included in the partition
        qubit_included = []
        if level == len(sequence):
            sequence.append([])

        for qubit in range(num_qubits):

            if i_stack[qubit]:
                gate = i_stack[qubit][0]
            else:
                continue

            # Check for dummy gate
            if is_measure_dummy(gate) or is_reset_dummy(gate):
                continue
            # Check for single gate
            elif is_single(gate):
                if qubit in qubit_included:
                    continue
                sequence[level].append(gate)
                qubit_included.append(qubit)
                i_set.remove(gate)      # Remove from Set
                i_stack[qubit].pop(0)   # Remove from Stack
            # Check for C-NOT gate
            elif is_double(gate):
                second_qubit = list(
                    set(gate.qubits).difference(set([qubit])))[0]
                buffer_gate = i_stack[second_qubit][0]

                # Checks if gate already included in the partition
                if qubit in qubit_included or second_qubit in qubit_included:
                    continue

                # Check if C-NOT is top in stacks of both of its indexes.
                if gate is buffer_gate:
                    qubit_included.append(qubit)
                    qubit_included.append(second_qubit)
                    sequence[level].append(gate)
                    i_set.remove(gate)
                    i_stack[qubit].pop(0)
                    i_stack[second_qubit].pop(0)
                # If not then don't add it.
                else:
                    continue
            # Check for unitary gate
            elif is_unitary(gate):
                gate_qubits = list(set(gate.qubits))
                qubit_already_used = (len(set(gate.qubits).intersection(qubit_included)) != 0)
                if qubit_already_used:
                    continue
                on_top_of_stack = True
                for q in gate_qubits:
                    ## "is" checks if variables point to same object in memory
                    ## while "==" checks if objects referred to by the variables are equal
                    ## Reference: https://stackoverflow.com/questions/132988/is-there-a-difference-between-and-is
                    on_top_of_stack = (gate is i_stack[q][0])
                    if not on_top_of_stack:
                        break
                if on_top_of_stack:
                    qubit_included.extend(gate_qubits)
                    sequence[level].append(gate)
                    i_set.remove(gate)
                    for q in gate_qubits:
                        i_stack[q].pop(0)

            elif is_measure(gate): 

                all_dummy = True
                for x in range(num_qubits):
                    if not i_stack[x]:
                        continue
                    # Intersection of both should be used
                    if not is_measure(i_stack[x][0]) and not is_measure_dummy(i_stack[x][0]):
                        all_dummy = False
                        break

                if all_dummy:
                    # Check if current level already has gates
                    if sequence[level]:
                        qubit_included = []
                        level += 1  # Increment the level
                        if level == len(sequence):
                            sequence.append([])

                    for x in range(num_qubits):
                        # Check if measure
                        if not i_stack[x]:
                            continue
                        if is_measure(i_stack[x][0]):
                            qubit_included.append(x)
                            sequence[level].append(i_stack[x][0])
                            # Remove from Instruction list
                            i_set.remove(i_stack[x][0])
                        i_stack[x].pop(0)
                    break  # To restart the Qubit loop from 0
            elif is_reset(gate):
                all_dummy = True
                for x in range(num_qubits):
                    if not i_stack[x]:
                        continue
                    # Intersection of both should be used
                    if not is_reset(i_stack[x][0]) and not is_reset_dummy(i_stack[x][0]):
                        all_dummy = False
                        break

                if all_dummy:
                    # Check if current level already has gates
                    if sequence[level]:
                        qubit_included = []
                        level += 1  # Increment the level
                        if level == len(sequence):
                            sequence.append([])

                    for x in range(num_qubits):
                        if not i_stack[x]:
                            continue
                        # Check if measure
                        if is_reset(i_stack[x][0]):
                            qubit_included.append(x)
                            sequence[level].append(i_stack[x][0])
                            # Remove from Instruction list
                            i_set.remove(i_stack[x][0])
                        i_stack[x].pop(0)
                    break  # To restart the Qubit loop from 0

            # Check if the instruction list is empty
            if not i_set:
                break

        level += 1
    return sequence, level

def partition(i_set, num_qubits):
    """ Partition the instruction set in to a number of levels.
        Levels have to be executed sequentially,
        while instructions within each level can be executed in parallel.
    Args:
        i_set (list): instruction set
        num_qubits (int): number of qubits
    Returns:
        partition_list (list): list of partitions
        levels (int): number of partitions
    """
    modified_i_set = []
    a = []
    for instruction in i_set:
        if instruction.name !='barrier':
            a.append(instruction)
        else:
            modified_i_set.append(a)
            a = []
    if a:
        modified_i_set.append(a)
    partition_list = []
    levels = 0
    for mod_ins in modified_i_set:
        if mod_ins != []:
            # Bell, Expect and Ensemble measure form a partitiom on their own.
            if mod_ins[0].name=='measure' and getattr(mod_ins[0],'params',None) != None and mod_ins[0].params[0] in ['Bell', 'Expect', 'Ensemble']:
                partition_list.append(mod_ins)
                levels += 1
            else:
                seq,level = partition_helper(mod_ins,num_qubits)
                partition_list.append(seq)
                levels += level
    partition_list = list(itertools.chain(*partition_list))

    return partition_list, levels    
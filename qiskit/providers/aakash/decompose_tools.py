from qiskit import *
from qiskit_aer import Aer
import numpy as np
import math, time, scipy, cmath, random, itertools, copy
from operator import itemgetter
from sympy.combinatorics import GrayCode
import qiskit.quantum_info as qi

# This file has been copied from gitrepo vr100/quantum-gd
# Reference:
# https://github.com/vr100/quantum-gd/blob/main/scripts/gate-decompose.py

STEPS = 1

def find_number_of_bits(size):
  n = math.log2(size)
  return math.ceil(n)

def get_gray_code(n):
  gcgen = GrayCode(n)
  gcs = list(gcgen.generate_gray())
  gc_list = [int(i) for gc in gcs for i in gc]
  gc_vector = np.array(gc_list).reshape(-1, n)
  return gc_vector

def get_binary_code(n):
  bc_len = int(math.pow(2,n))
  fmt = f"{{0:0{n}b}}"
  bcs = [fmt.format(i) for i in range(bc_len)]
  bc_list = [int(i) for bc in bcs for i in bc]
  bc_vector = np.array(bc_list).reshape(-1, n)
  return bc_vector

# Note that the control bit for cx is the position
# of change between the gray code of current index
# and that of next index
def get_ctrl_bit_for_cx(ctrl_bits, gcodes, cur_idx):
  next_idx = cur_idx+1 if (cur_idx+1) < len(gcodes) else 0
  diff = abs(gcodes[next_idx] - gcodes[cur_idx])
  idx = list(diff).index(1)
  return ctrl_bits[idx]

# implementation of multiplexed rk (or uniformly
# controlled rk gates) using the following papers
# reference:
# "Transformation of quantum states using uniformly controlled rotations"
# https://arxiv.org/abs/quant-ph/0407010
# "Synthesis of Quantum Logic Circuits"
# https://arxiv.org/abs/quant-ph/0406176
def get_gates_for_crk(list_rk, bits, k):
  if len(list_rk) <= 1:
    return list_rk
  if not k in ["y","z"]:
    print(f"Unsupported Rk where k = {k}, exiting...")
    exit(-1)
  rk = "ry" if k == "y" else "rz"
  rkbit, ctrl_bits = bits[-1], bits[:-1]
  nc = len(ctrl_bits)
  bcodes = get_binary_code(nc)
  gcodes = get_gray_code(nc)
  alphas = [rk["params"][0] for rk in list_rk]
  r = int(math.pow(2, nc))
  M = [ (1/r) * math.pow(-1, np.dot(bcodes[j], gcodes[i]))
    for i in range(r) for j in range(r)]
  M = np.array(M).reshape(r,r)
  thetas = M @ np.array(alphas)
  gates = []
  for i in range(len(thetas)):
    gate_rk = {"oper": rk, "params":[thetas[i]], "bits":[rkbit]}
    cxbit = get_ctrl_bit_for_cx(ctrl_bits, gcodes, i)
    gate_cx = {"oper": "cx", "params":[], "bits":[cxbit, rkbit]}
    gates.extend([gate_rk, gate_cx])
  return gates

# Note that 0 is the least significant bit
# for qiskit
def get_gates_for_crz(list_rz, nbits, sbit):
  bits = list(range(sbit, sbit+nbits))[::-1]
  return get_gates_for_crk(list_rz, bits, k="z")

# d is of form
# [p 0]  =  [e^(-ix) 0]
# [0 q]     [0 e^(-iy)]
# thus x = i lnp, y = i lnq
#
# find a,b such that
# [e^(-ix) 0] = [e^(ia) 0] *  [e^(-ib) 0]
# [0 e^(-iy)]   [0 e^(ia)]    [0  e^(ib)]
# thus d = Ph(a)Rz(2b)
# where Ph is global phase gate
#
# solving above, we get
# a = -(x+y)/2
# b = (x-y)/2
#
def get_a_b_angles(d):
  p,q = d[0,0], d[1,1]
  if p == 0 or q == 0: return 0,0
  x = 1j * cmath.log(p)
  y = 1j * cmath.log(q)
  a = -(x+y)/2.0
  b = (x-y)/2.0
  return a,b

def eye(size):
  return np.eye(size, dtype=np.cdouble)

# diagonal gate expands to multiplexed rz gate
# and another diagonal gate (composed of the
# phase angles)
def get_gates_for_diagonal(d, sbit=0):
  r,_ = d.shape
  if r == 2:
    a,b = get_a_b_angles(d)
    phase = {"oper": "ph", "params": [a], "bits": [sbit]}
    rz = {"oper": "rz", "params": [2*b], "bits": [sbit]}
    gates = [rz, phase]
    return gates
  if np.allclose(eye(r), d):
    return []
  nbits = find_number_of_bits(r)
  list_rz, list_phase = [],[]
  for i in range(r//2):
    start = 2*i
    end = start + 2
    sub = d[start:end,start:end]
    rz,phase = get_gates_for_diagonal(sub)
    list_rz.append(rz)
    list_phase.append(phase)
  phases = [cmath.exp(1j * item["params"][0])
    for item in list_phase]
  d_phases = np.diag(phases)
  phase_gates = get_gates_for_diagonal(d_phases, sbit=sbit+1)
  rz_gates = get_gates_for_crz(list_rz, nbits, sbit)
  gates = [*rz_gates, *phase_gates]
  return gates

def get_gates_for_cry(list_rys, bits):
  return get_gates_for_crk(list_rys, bits, k="y")

# ry is on most significant bit (msb),
# all other bits are control bits
def demultiplex_ry_by_lsb(cs):
  r,_ = cs.shape
  angles = []
  for i in range(r//2):
    a = 2 * math.acos(cs[i,i])
    angles.append(a)
  return angles

def multiplex_ry_by_lsb(angles):
  angles = np.array(angles)
  cosa = np.diag(np.cos(angles))
  sina = np.diag(np.sin(angles))
  sina_ = np.diag(-np.sin(angles))
  blocks = [[cosa, sina_],[sina, cosa]]
  return np.bmat(blocks)

# cs is a list of list of angles
# for eg: cs = [(a1, a2), (a3, a4)]
# where multi = 1 (i.e, second level of csd decomposition)
# for a matrix of size 8x8
def multiplex_central_matrix(cs, multi):
  lst = []
  for item in cs:
    mat = multiplex_ry_by_lsb(item)
    lst.append(mat)
  cs_mat = multiplex_msb_single(lst, multi)
  return cs_mat

# for multi = 1,
# cs is also of form
# [C0 0]
# [0 C1]
# len(ctrl-bits) = multi
# so for multi >=1, we need to demultiplex
# note the control bits here are the most
# significant bits
#
# each demultiplexed central matrix (C0,C1...)
# is part of controlled ry where the control
# bits are the least significant bits
#
def demultiplex_ry(cs, multi):
  rymat = demultiplex_msb(cs, multi)
  r,_ = cs.shape
  # multi is the number of ctrl bits in the msb
  # then the ry qubit
  # the remaining lsb (if any) are also ctrl bits
  nbits = find_number_of_bits(r)
  allbits = list(range(0,nbits))[::-1]
  rybit = allbits[multi]
  ctrl_bits = [*allbits[0:multi],*allbits[multi+1:]]
  bits = [*ctrl_bits, rybit]
  list_rys = []
  for rym in rymat:
    angles = demultiplex_ry_by_lsb(rym)
    for ai in angles:
      ry = {"oper": "ry", "params": [ai], "bits": bits}
      list_rys.append(ry)
  return list_rys, bits

# cs contains a list of list of angles
# for eg: cs = [(a1, a2), (a3, a4)]
# where multi = 1 (i.e, second level of csd decomposition)
# for a matrix of size 8x8
def get_gates_for_center_matrix(cs, multi):
  if len(cs) == 0:
    return []
  r = 2 * len(cs) * len(cs[0])
  nbits = find_number_of_bits(r)
  allbits = list(range(0,nbits))[::-1]
  rybit = allbits[multi]
  ctrl_bits = [*allbits[0:multi],*allbits[multi+1:]]
  bits = [*ctrl_bits, rybit]
  list_rys = []
  for angles in cs:
    for ai in angles:
      ry = {"oper": "ry", "params": [2*ai], "bits": bits}
      list_rys.append(ry)
  gates = get_gates_for_cry(list_rys, bits)
  return gates

# cosine sine decomposition (csd)
# u,cs,vh = csd(U)
# for a given unitary matrix U
# u,vh are of form:
# [X1 0]
# [0 X2]
# X1, X2 are unitaries multiplexed by the most
# significant bit(s)
#
def demultiplex_msb(X, multi=0):
  if multi == 0:
    return [X]
  r,c = X.shape
  ssize = math.pow(2, multi)
  if r < ssize or r % ssize != 0 :
    print(f"matrix of size {r,c} cannot be mutliplexed by {multi} qubits")
    print(f"exiting...")
    exit(-1)
  t = [block for rowblock in np.vsplit(X, ssize)
    for block in np.hsplit(rowblock, ssize)]
  sr = (int)(r // ssize)
  z = np.zeros((sr,sr))
  unitaries = []
  for item in t:
    if not np.allclose(item, z):
      unitaries.append(item)
  return unitaries

def multiplex_msb_single(lst, multi):
  bcount, t = len(lst), []
  for i in range(bcount):
    u = lst[i]
    z = np.zeros(u.shape)
    h = [z] * bcount
    h[i] = u
    t.append(h)
  return np.bmat(t)

def get_csd_for_unitary(U, debug=False):
  r,_= U.shape
  p = r//2
  try:
    u, cs, vh = scipy.linalg.cossin(U, p=p, q=p, separate=True)
    return u,cs,vh
  except Exception as e:
    print(f"cosine sine decomposition exception: {e}")
    if debug:
      # pretty print rounds off very small values (< 1e-200)
      # which causes the issue, so using normal print
      print(f"matrix: {U}")
    # issue with a specific matrix where
    # really small values [smaller than 1e-200] are
    # present, so rounding them off to zero
    nU = np.where(np.abs(U) < 1e-50, 0.0, U)
    u, cs, vh = scipy.linalg.cossin(nU, p=p, q=p, separate=True)
    return u,cs,vh

def is_diagonal(x):
  non_zeros = np.count_nonzero(x - np.diag(np.diagonal(x)))
  return non_zeros == 0

def get_gates_for_multiplexed_unitary(mU, multi=0, debug=False):
  diagonal = True
  for item in mU:
    diagonal = diagonal and is_diagonal(item)
  if diagonal:
    U = multiplex_msb_single(mU, multi)
    return get_gates_for_diagonal(U)
  list_u,list_cs,list_vh = [],[],[]
  for item in mU:
    u,cs,vh = get_csd_for_unitary(item)
    list_u.extend(u)
    list_cs.append(cs)
    list_vh.extend(vh)
  gates_u = get_gates_for_multiplexed_unitary(list_u, multi+1, debug)
  gates_vh = get_gates_for_multiplexed_unitary(list_vh, multi+1, debug)
  gates_cs = get_gates_for_center_matrix(list_cs, multi)
  if debug:
    u = multiplex_msb_single(list_u, multi+1)
    vh = multiplex_msb_single(list_vh, multi+1)
    cs = multiplex_central_matrix(list_cs, multi+1)
    verify_gates(gates_u, u, vtype="all", name=f"u_{multi}")
    verify_gates(gates_vh, vh, vtype="all", name=f"vh_{multi}")
    verify_gates(gates_cs, cs, vtype="all", name=f"cs_{multi}")
  # Note the order of gates is inverse of the
  # order of multiplication
  # U = u @ cs @ vh
  # hence gates have to be in inverse order (vh, cs, u)
  return [*gates_vh, *gates_cs, *gates_u]

def is_unitary(x):
  m = np.matrix(x)
  return np.allclose(eye(m.shape[0]), m.H * m)

def get_gates_for_unitary(U):
  r,c = U.shape
  if r !=c or r % 2 != 0:
    print(f"matrix of size {r},{c} is not supported")
    print("exiting...")
    exit(-1)
  res = is_unitary(U)
  #print(f"unitarity: {res}")
  return get_gates_for_multiplexed_unitary([U])

# cx gate is a two qubit gate
# and may be non neighbouring...
# the cx_gate function computes the
# matrix for the range of control bit
# and actual bit
#
# for eg, [0, 1] computes cx for bits 0 and 1
# ctrl-bit: 0, actual-bit: 1
# while [3, 0] computes cx from 0 to 3 (also
# incorporating bits 1 and 2 in the computed matrix)
# ctrl-bit: 3, actual-bit: 0
# also, 0 is least significant bit here
#
# Note: for bits q2,q1,q0 - order of kron is
# np.kron(q2, np.kron(q1, q0)) where q0 is the
# least significant bit
def cx_gate(bits):
  ctrl_bit = bits[0]
  qbit = bits[1]
  start = min(ctrl_bit, qbit)
  end = max(ctrl_bit, qbit) + 1
  i2 = eye(2)
  x = np.array([[0, 1], [1, 0]], dtype=np.cdouble)
  data = eye(1)
  for i in range(start, end):
    if i == ctrl_bit:
      continue
    if i == qbit:
      data = np.kron(x, data)
    else:
      data = np.kron(i2, data)
  # Note: the qiskit documentation for CXGate
  # helps understand the logic below
  # https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.CXGate
  # if ctrl_bit is more significant than qbit
  # cx = |0><0| x I + |1><1| x data
  # else
  # cx = I x |0><0| + data x |1><1|
  v0 = np.array([1, 0], dtype=np.cdouble)
  v1 = np.array([0, 1], dtype=np.cdouble)
  v00 = np.outer(v0,v0)
  v11 = np.outer(v1, v1)
  neye = int(math.pow(2, abs(qbit-ctrl_bit)))
  if ctrl_bit > qbit:
    cx = np.kron(v00, eye(neye)) + np.kron(v11, data)
  else:
    cx = np.kron(eye(neye), v00) + np.kron(data, v11)
  return cx

def get_global_phase_gate(angle):
  factor = cmath.exp(1j * angle)
  return factor * eye(2)

# using custom ry, rz gate for complex numbers
def rygate(theta):
  angle = theta/2.0
  cosa = cmath.cos(angle)
  sina = cmath.sin(angle)
  return np.array([
    [cosa, -sina],
    [sina, cosa]
  ], dtype=np.cdouble)

def rzgate(phi):
  angle = phi/2.0
  ei_a = cmath.exp(-1j * angle)
  eia = cmath.exp(1j * angle)
  return np.array([
    [ei_a, 0.0],
    [0.0, eia]
  ], dtype=np.cdouble)

def get_oper_matrix(oper, params, bits):
  if oper == "ry":
    return rygate(params[0])
  if oper == "rz":
    return rzgate(params[0])
  if oper == "cx":
    return cx_gate(bits)
  if oper == "ph":
    return get_global_phase_gate(params[0])
  print(f"unsupported operation: {oper}, exiting...")
  exit(-1)

def get_skip_till(bits):
  if len(bits) == 1:
    return -1
  return max(bits[0],bits[1])

def get_gate_matrix(gate, nqubits):
  oper, params, bits = itemgetter(
    "oper", "params", "bits")(gate)
  omat = get_oper_matrix(oper, params, bits)
  gmat = eye(1)
  skip_till = -1
  for i in range(nqubits):
    if i <= skip_till:
      continue
    if i in bits:
      gmat = np.kron(omat, gmat)
      skip_till = get_skip_till(bits)
    else:
      gmat = np.kron(eye(2), gmat)
  return gmat

def gate_str(gate):
  oper, params, bits = itemgetter("oper",
      "params", "bits")(gate)
  return f"{oper}({params}) {bits}"

def pretty_print_gates(gates, name):
  print(f"gates for {name}:")
  for gate in gates:
    print(gate_str(gate))

def pretty_print_matrix(U, name):
  print(f"matrix for {name}")
  print(f"matrix: {np.around(U, decimals=3)}")

def verify_same(exp_qc, actual_qc, name="", debug=False):
  eU = qi.Operator(exp_qc).data
  aU = qi.Operator(actual_qc).data
  result = np.allclose(eU, aU)
  if debug:
    print(f"### debug details for {name} ###")
    pretty_print_matrix(eU, name=f"{name}_expected")
    pretty_print_matrix(aU, name=f"{name}_actual")
    if not result:
      print(np.isclose(eU, aU))
  if name:
    print(f"circuit similarity for {name}: {result}")
  return result

def verify_gates(gates, U, vtype="matrix", name="", debug=False):
  result, qresult = None, None
  r,_ = U.shape
  nqubits = find_number_of_bits(r)
  if vtype == "all" or vtype == "matrix":
    cU = eye(r)
    for gate in gates:
      gmat = get_gate_matrix(gate, nqubits)
      # note that the matrix multiplication order
      # and circuit gate order are inverse
      cU = gmat @ cU
    result = np.allclose(U, cU)
    if debug:
      print(f"### debug details for {name} ###")
      pretty_print_gates(gates, name=f"{name}_gates")
      pretty_print_matrix(U, name=f"{name}_original")
      pretty_print_matrix(cU, name=f"{name}_generated")
  if vtype == "all" or vtype == "circuit":
    qc = QuantumCircuit(nqubits)
    qc = build_gates(qc, gates)
    qcU = qi.Operator(qc).data
    qresult = np.allclose(U, qcU)
  if name:
    print(f"verification for {name} with type '{vtype}' " + \
      f"[circuit]: {qresult}, [matrix]: {result}")
  return result

def verify_circuit(qc, U, name="", debug=False):
  qcU = qi.Operator(qc).data
  qresult = np.allclose(U, qcU)
  if name:
    print(f"circuit verification for {name}: {qresult}")
  if debug:
    print(f"### debug details for {name} ###")
    pretty_print_matrix(U, name=f"{name}_original")
    pretty_print_matrix(qcU, name=f"{name}_circuit")
    if not qresult:
      print("Closeness: ")
      print(np.isclose(U,qcU))
  if not qresult:
    print(f"Circuit and matrix does not match for {name}, exiting...")
    exit(-1)
  return qresult

def build_gates(qc, gates):
  phase = 0
  for gate in gates:
    oper, params, bits = itemgetter(
      "oper", "params", "bits")(gate)
    if oper == "cx":
      qc.cx(bits[0], bits[1])
    elif oper == "ry":
      if np.isclose(params[0].imag, 0.0):
        qc.ry(params[0].real, bits[0])
      else:
        u_ry = get_oper_matrix(oper, params, bits)
        qc.unitary(u_ry, bits)
    elif oper == "rz":
      if np.isclose(params[0].imag, 0.0):
        qc.rz(params[0].real, bits[0])
      else:
        u_rz = get_oper_matrix(oper, params, bits)
        qc.unitary(u_rz, bits)
    elif oper == "ph":
      if np.isclose(params[0].imag, 0.0):
        phase += params[0].real
      else:
        u_ph = get_oper_matrix(oper, params, bits)
        qc.unitary(u_ph, bits)
  qc.global_phase += phase
  return qc

def build_decompose(U):
  r,_ = U.shape
  nbits = find_number_of_bits(r)
  gates = get_gates_for_unitary(U)
  print(f"gate count: {len(gates)}")
  verify_gates(gates, U, name=f"decompose-gates-{nbits}")
  data = QuantumRegister(nbits)
  qc = QuantumCircuit(data)
  basic_qc = build_gates(qc, gates)
  verify_circuit(basic_qc, U, name=f"decompose-gates-{nbits}")
  return basic_qc

def execute_circuit(qc):
  backend = Aer.get_backend('statevector_simulator')
  qc = transpile(qc, backend=backend)
  job_sim = backend.run(qc)
  result_sim = job_sim.result()
  sv = result_sim.get_statevector(qc)
  probs = sv.probabilities_dict()
  threshold = 1e-6
  probs = {k:v for k,v in probs.items() if v > threshold}
  print(f"result: {probs}")

def repeat_gates(qc, steps=STEPS):
  return qc.repeat(steps)

def decompose_gate(U):
  qc = build_decompose(U)
  qc = repeat_gates(qc)
  execute_circuit(qc)

def fix_phase_if_required(qc, tqc, U):
  qU = qi.Operator(tqc).data
  factor = qU[0,0] / U[0,0]
  lam = 1j * cmath.log(factor)
  fqc = copy.deepcopy(tqc)
  fqc.global_phase = fqc.global_phase + lam.real
  result = verify_same(qc, fqc, name="phasefix")
  if not result:
    verify_same(qc, tqc, name="transpile_debug", debug=True)
    print("phase fix also did not help, exiting...")
    exit(-1)
  return fqc

def remove_no_action_gates(qc):
  tqc = copy.deepcopy(qc)
  data, count = [], 0
  for item in qc:
    oper, params = item.operation.name, item.operation.params
    if (oper == "rz" or oper == "ry") and params[0] == 0:
      count += 1
    else:
      data.append(item)
  tqc.data = data
  print(f"no-action gates removed: {count}")
  return tqc

def transpile_circuit(qc, U):
  rqc = remove_no_action_gates(qc)
  result = verify_same(qc, rqc, name="remove-no-action-gates")
  tqc = transpile(qc, basis_gates=['rx', 'ry', 'cx', 'rz'],
    optimization_level=3)
  result = verify_same(qc, tqc, name="transpile")
  if not result:
    tqc = fix_phase_if_required(qc, tqc, U)
  return tqc

def print_ops(qc, name=""):
  print(f"ops for {name}: {qc.count_ops()}")

def decompose_and_transpile(U):
  qc = build_decompose(U)
  print_ops(qc, name="decompose")
  qc = transpile_circuit(qc, U)
  print_ops(qc, name="transpile")
  qc = repeat_gates(qc)
  print_ops(qc.decompose(), name="repetition")
  execute_fn = lambda: execute_circuit(qc)
  execute_and_measure(execute_fn, "executing circuit")

def execute_and_measure(test_fn, fn_name):
  start = time.time()
  test_fn()
  end = time.time()
  print(f"time taken for {fn_name} is {end - start} seconds")

def _get_matrix(oper):
  if oper == "i":
    return np.array([[1.0, 0.0], [0.0, 1.0]])
  if oper == "x":
    return np.array([[0.0, 1.0], [1.0, 0.0]])
  if oper == "y":
    return np.array([[0.0, complex(0.0, -1.0)],
      [complex(0.0, 1.0), 0.0]])
  if oper == "z":
    return np.array([[1.0, 0.0], [0.0, -1.0]])

def _get_unitary_matrix(opers, value):
  cos_matrix = np.array([1.0], dtype=np.cdouble)
  sin_matrix = np.array([1.0], dtype=np.cdouble)
  for o in opers:
    cos_matrix = np.kron(np.eye(2), cos_matrix)
    sin_matrix = np.kron(_get_matrix(o), sin_matrix)
  u_matrix = math.cos(value) * cos_matrix + \
    (-1j) * math.sin(value) * sin_matrix
  return u_matrix

def get_input(opers, nbits, steps=STEPS):
  value = math.pow(2, -1 * nbits) * (1/steps)
  return _get_unitary_matrix(opers, value)

OPERS = ['i', 'x', 'y', 'z']
BITS = range(1, 8)
CHOICES = ["all", "mirrored"]

def get_random_input():
  n = random.choice(BITS)
  choice = "all"
  if n % 2 == 0:
    choice = random.choice(CHOICES)
  if choice == "mirrored":
    print(f"mirroring opers...")
    perms = list(itertools.product(OPERS, repeat=n//2))
    opers = random.choice(perms)
    opers = [*opers, *opers[::-1]]
  else:
    perms = list(itertools.product(OPERS, repeat=n))
    opers = random.choice(perms)
  return get_input(opers, n), n, opers

def get_fixed_input():
  # define fixed opers list
  # e.g. opers = ['x', 'i', 'i', 'i', 'i', 'x']
  n = len(opers)
  return get_input(opers, n), n, opers

def main():
  U,n,opers = get_random_input()
  print(f"bit count: {n}")
  print(f"input opers: {opers}")
  decompose_fn = lambda : decompose_and_transpile(U)
  execute_and_measure(decompose_fn,
    f"decomposing {n} qubit unitary for {STEPS} steps")

if __name__ == "__main__":
  main()

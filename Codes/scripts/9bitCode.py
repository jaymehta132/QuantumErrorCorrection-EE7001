from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

results_dir = 'results/9bitCode/'
import os
os.makedirs(results_dir, exist_ok=True)
# --------------------------
# Parameters
# --------------------------
ERROR_QUBIT = 1  # Inject Z error on this qubit (0..8)
FINAL_SHOTS = 1024

# --------------------------
# Qubit and classical registers
# --------------------------
q = QuantumRegister(9, 'q')          # data qubits
a = QuantumRegister(6, 'a')          # ancillas for syndrome
syn = ClassicalRegister(6, 'syn')    # syndrome measurement bits
final_c = ClassicalRegister(1, 'final')  # final logical qubit

qc = QuantumCircuit(q, a, syn, final_c)

# --------------------------
# Encode |+> as logical qubit
# --------------------------
def encode_shor(qc, q):
    # Start with |+> on q0
    qc.h(q[0])
    # Outer repetition (phase-flip)
    qc.cx(q[0], q[3])
    qc.cx(q[0], q[6])
    qc.h(q[0]); qc.h(q[3]); qc.h(q[6])
    # Inner 3-qubit bit-flip codes
    qc.cx(q[0], q[1]); qc.cx(q[0], q[2])
    qc.cx(q[3], q[4]); qc.cx(q[3], q[5])
    qc.cx(q[6], q[7]); qc.cx(q[6], q[8])

def decode_shor(qc, q):
    # Inverse of inner 3-qubit bit-flip codes
    qc.cx(q[6], q[8]); qc.cx(q[6], q[7])
    qc.cx(q[3], q[5]); qc.cx(q[3], q[4])
    qc.cx(q[0], q[2]); qc.cx(q[0], q[1])
    # Inverse outer phase-flip basis conversion
    qc.h(q[6]); qc.h(q[3]); qc.h(q[0])
    # Inverse outer repetition
    qc.cx(q[0], q[6]); qc.cx(q[0], q[3])


encode_shor(qc, q)

# --------------------------
# Inject Z error
# --------------------------
qc.barrier()
qc.z(q[ERROR_QUBIT])
qc.barrier()

# --------------------------
# Z-syndrome measurement (phase-flip detection)
# Use H basis rotation so Z errors become detectable as X flips
# Block 1: q0,q1,q2 -> anc a0,a1
for i in [0,1,2]:
    qc.h(q[i])
qc.cx(q[0], a[0]); qc.cx(q[1], a[0]); qc.measure(a[0], syn[0]); qc.reset(a[0])
qc.cx(q[1], a[1]); qc.cx(q[2], a[1]); qc.measure(a[1], syn[1]); qc.reset(a[1])
for i in [0,1,2]:
    qc.h(q[i])

# Block 2: q3,q4,q5 -> anc a2,a3
for i in [3,4,5]:
    qc.h(q[i])
qc.cx(q[3], a[2]); qc.cx(q[4], a[2]); qc.measure(a[2], syn[2]); qc.reset(a[2])
qc.cx(q[4], a[3]); qc.cx(q[5], a[3]); qc.measure(a[3], syn[3]); qc.reset(a[3])
for i in [3,4,5]:
    qc.h(q[i])

# Block 3: q6,q7,q8 -> anc a4,a5
for i in [6,7,8]:
    qc.h(q[i])
qc.cx(q[6], a[4]); qc.cx(q[7], a[4]); qc.measure(a[4], syn[4]); qc.reset(a[4])
qc.cx(q[7], a[5]); qc.cx(q[8], a[5]); qc.measure(a[5], syn[5]); qc.reset(a[5])
for i in [6,7,8]:
    qc.h(q[i])

# --------------------------
# Run single-shot syndrome to determine corrections
# --------------------------
sim = AerSimulator()
t_qc = transpile(qc, sim)
job = sim.run(t_qc, shots=1, memory=True)
res = job.result()
mem = res.get_memory()[0][::-1]  # reverse so index i corresponds to syn[i]
mem = ''.join(filter(lambda x: x in '01', res.get_memory()[0]))  # keep only '0' or '1'
syndrome_bits = [int(b) for b in mem[::-1]]

print("Syndrome bits:", syndrome_bits)

# --------------------------
# Classical decoding: determine which qubits to correct
# --------------------------
corrections = []

# Block 1: q0,q1,q2 -> syn[0], syn[1]
m1, m2 = syndrome_bits[0], syndrome_bits[1]
if (m1,m2)==(1,0): corrections.append(0)
elif (m1,m2)==(1,1): corrections.append(1)
elif (m1,m2)==(0,1): corrections.append(2)

# Block 2: q3,q4,q5 -> syn[2], syn[3]
m1, m2 = syndrome_bits[2], syndrome_bits[3]
if (m1,m2)==(1,0): corrections.append(3)
elif (m1,m2)==(1,1): corrections.append(4)
elif (m1,m2)==(0,1): corrections.append(5)

# Block 3: q6,q7,q8 -> syn[4], syn[5]
m1, m2 = syndrome_bits[4], syndrome_bits[5]
if (m1,m2)==(1,0): corrections.append(6)
elif (m1,m2)==(1,1): corrections.append(7)
elif (m1,m2)==(0,1): corrections.append(8)

print("Corrections to apply:", corrections)

# --------------------------
# Build final circuit: encode |+>, apply Z error, apply corrections, decode
# --------------------------
q2 = QuantumRegister(9, 'q2')
final_c = ClassicalRegister(1, 'final')
final_qc = QuantumCircuit(q2, final_c, name='final')

encode_shor(final_qc, q2)
final_qc.barrier()
final_qc.z(q2[ERROR_QUBIT])
final_qc.barrier()
for qubit in corrections:
    final_qc.z(q2[qubit])
final_qc.barrier()
decode_shor(final_qc, q2)

# Measure logical qubit in X basis to verify |+> → |0> after correction
final_qc.h(q2[0])
final_qc.measure(q2[0], final_c[0])

print(final_qc.draw('text'))
print(final_qc.draw('mpl', filename=results_dir+'FinalCircuit.png'))

# --------------------------
# Run simulation
# --------------------------
t_final = transpile(final_qc, sim)
job2 = sim.run(t_final, shots=FINAL_SHOTS)
res2 = job2.result()
counts = res2.get_counts()

print("\nFinal measurement counts after correction (|+> → |0>):")
print(counts)
success = counts.get('0',0)/FINAL_SHOTS
print(f"Success probability (should be 1.0): {success:.4f}")

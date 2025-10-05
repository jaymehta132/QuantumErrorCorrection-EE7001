from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import os

results_dir = 'results/3bitCode/'

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)
# 5 qubits: 3 data + 2 ancilla
qr = QuantumRegister(5, 'q')

# Classical registers
cr_syndrome = ClassicalRegister(2, 'syndrome')  # For ancilla measurement
cr_logical = ClassicalRegister(1, 'logical')    # For final logical qubit
qc = QuantumCircuit(qr, cr_syndrome, cr_logical)

# Encoding
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.barrier(label="Encoding")

# Error
qc.x(1)
qc.barrier(label="Error Injection")

# Syndrome extraction
qc.cx(0, 3)
qc.cx(1, 3)
qc.cx(0, 4)
qc.cx(2, 4)

qc.measure([3, 4], cr_syndrome)  # measure ancillas into syndrome register
qc.barrier(label="Detection")
# Correction based on syndrome
with qc.if_test((cr_syndrome, 0)):
    pass  # syndrome=00 → no error
with qc.if_test((cr_syndrome, 1)):
    qc.x(2)  # syndrome=01 → flip qubit 2
with qc.if_test((cr_syndrome, 2)):
    qc.x(1)  # syndrome=10 → flip qubit 1
with qc.if_test((cr_syndrome, 3)):
    qc.x(0)  # syndrome=11 → flip qubit 0
qc.barrier(label="Correction")

# Decode
qc.cx(0, 1)
qc.cx(0, 2)
qc.h(0)

# Measure logical qubit
qc.measure(0, cr_logical[0])

# Simulate
sim = AerSimulator()
# result = execute(qc, sim, shots=1024).result()
result = sim.run(qc, shots=1024).result()
counts = result.get_counts()

# Output
print("Measurement counts:", counts)
print(qc.draw('mpl', filename=f'{results_dir}3bitCodeCircuit.png'))
plt.show()
plot_histogram(counts)
plt.savefig(f'{results_dir}3bitCodeHistogram.png')
plt.show()
plt.close()

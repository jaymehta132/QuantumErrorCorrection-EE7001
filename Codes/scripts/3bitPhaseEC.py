import numpy as np
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit import QuantumCircuit, Parameter

results_dir = 'results/3bitPhaseEC/'
import os
os.makedirs(results_dir, exist_ok=True)
# --- 1. SETUP PARAMETERS AND INITIAL STATE ---

# Define the small error parameter epsilon (e.g., 0.1 radians)
epsilon = 0.1
# The fidelity of the unencoded state should be approximately 1 - epsilon^2
# The fidelity of the encoded, corrected state should be approximately 1 - epsilon^6

# Define the initial single-qubit state |0> (qubit 0).
# We choose |0> because it is NOT an eigenstate of the X error, which allows the fidelity
# suppression effect to be visible, matching the paper's derivation.
initial_state = Statevector([1, 0]) # State |0>

# --- 2. UNENCODED FIDELITY CALCULATION (Reference) ---

# The error operator on a single qubit is U = exp(i*epsilon*X)
# U = cos(epsilon) * I + i*sin(epsilon) * X
U_error = np.array([
    [np.cos(epsilon), 1j * np.sin(epsilon)],
    [1j * np.sin(epsilon), np.cos(epsilon)]
])
U_operator = Operator(U_error)

# Apply the error to the initial state
final_unencoded_state = initial_state.evolve(U_operator)

# Calculate the unencoded fidelity: F = |<psi|U|psi>|^2
# For |psi>=|0>, F = |<0|U|0>|^2 = |cos(epsilon)|^2
Funencoded = np.abs(initial_state.inner(final_unencoded_state))**2

print(f"--- Quantum Error Correction (3-Qubit Bit-Flip Code) ---")
print(f"Error Magnitude (epsilon): {epsilon}")
print(f"Unencoded Fidelity F_unencoded = |<0|U|0>|^2:")
print(f"Calculated: {Funencoded:.8f}")
print(f"Approximation (1 - epsilon**2): {1 - epsilon**2:.8f}\n")


# --- 3. ENCODED FIDELITY CALCULATION ---

# System layout: 3 data qubits (q0, q1, q2) + 2 ancilla qubits (a0, a1). Total 5 qubits.

# A. Encoding Circuit (U_enc): |psi> -> |psi>_L |00>_A
# We encode |0> -> |000>
qc_enc = QuantumCircuit(3)
# The initial state is implicitly |0>, so no H gate is applied to encode to |000>.
qc_enc.cx(0, 1)
qc_enc.cx(0, 2)

# Initial encoded state |psi>_L = |000>
initial_encoded_state = Statevector(qc_enc)

# Initial state for the 5-qubit system: |psi>_L |00>_A
# The statevector ordering is LSB (q0) to MSB (a1): |q0 q1 q2 a0 a1>
state_L_A = initial_encoded_state.tensor(Statevector.from_label('00'))
# Qubit indices: q0, q1, q2 (data, indices 0, 1, 2); a0, a1 (ancilla, indices 3, 4)


# B. Error Operator (E): U_error tensor U_error tensor U_error tensor I tensor I
I_op = Operator.from_label('I')
U_op = Operator(U_error)

# Construct the 5-qubit operator in MSB (I_a1) to LSB (U_q0) order.
E_operator = I_op.tensor(I_op).tensor(U_op).tensor(U_op).tensor(U_op)

# Apply the error
state_after_error = state_L_A.evolve(E_operator)
print("--- 5-Qubit Error Circuit (E) ---")
print(qc_enc.draw('text'))
print(qc_enc.draw('mpl', filename='results/3bitPhaseEC/EncodingCircuit.png'))
print("-" * 50)


# C. QEC Unitary (U_QEC) - Syndrome Measurement
# Stabilizers: Z0Z1 (measured by a0), Z1Z2 (measured by a1)
# q0, q1, q2 are data; q3 (a0), q4 (a1) are ancilla
qc_qec = QuantumCircuit(5, name="Syndrome_Meas")
# Measure Z0Z1 using ancilla a0 (q3)
qc_qec.cx(0, 3)
qc_qec.cx(1, 3)
# Measure Z1Z2 using ancilla a1 (q4)
qc_qec.cx(1, 4)
qc_qec.cx(2, 4)
qc_qec.barrier()
UQEC_operator = Operator(qc_qec)

# Print the circuit diagram for visualization
print("--- 5-Qubit Syndrome Measurement Circuit (U_QEC) ---")
print(qc_qec.draw('text'))
print(qc_qec.draw('mpl', filename='results/3bitPhaseEC/SyndromeMeasurementCircuit.png'))
print("-" * 50)

# Apply the QEC circuit
state_after_QEC = state_after_error.evolve(UQEC_operator)


# D. Projection onto the 'No Error Detected' Subspace (Syndrome 00)
# We look for the component where ancilla qubits (indices 3 and 4) are in state |00>.
# The state is ordered |q0 q1 q2 a0 a1> (LSB to MSB)
# In the 32-element vector, the first 8 elements correspond to |a0 a1> = |00> (indices 0 to 7)

# We extract the components where a1=0 and a0=0 (indices 0 to 7)
no_error_component_vector = np.zeros(32, dtype=complex)
# The 00-component corresponds to the first 8 elements (0 to 7)
no_error_component_vector[0:8] = state_after_QEC.data[0:8]

# Normalize the component vector to get the state after post-selection
norm_sq_00 = np.sum(np.abs(no_error_component_vector)**2)
P_00 = norm_sq_00 # Probability of measuring 00
# Handle the case where norm_sq_00 is zero to prevent division by zero
if norm_sq_00 > 1e-12:
    state_post_selection = Statevector(no_error_component_vector / np.sqrt(norm_sq_00))
else:
    state_post_selection = Statevector(no_error_component_vector)
    print("Warning: Probability of '00' syndrome is nearly zero.")


# E. Decoding and Final Fidelity
# The 00-syndrome correction operator is I (Identity).
# We must now decode the 3-qubit state back to a single qubit.
qc_dec = QuantumCircuit(3)
qc_dec.cx(0, 2)
qc_dec.cx(0, 1)
# We simply apply the inverse of the encoding circuit.
Udec_data = Operator(qc_dec)

print("--- 3-Qubit Decoding Circuit (U_dec) ---")
print(qc_dec.draw('text'))
print(qc_dec.draw('mpl', filename='results/3bitPhaseEC/DecodingCircuit.png'))
print("-" * 50)

# Extract the 3-qubit data state (indices 0, 1, 2)
# The data component is the 8-element vector we extracted for the |00> syndrome.
unnormalized_data_state = no_error_component_vector[0:8]
norm_data_sq = np.sum(np.abs(unnormalized_data_state)**2)
data_state_corrected_L = Statevector(unnormalized_data_state / np.sqrt(norm_data_sq))

# Apply the decoding operator (only on the data qubits)
final_decoded_state = data_state_corrected_L.evolve(Udec_data)

# The ideal final state is the unencoded state, |0> |00> in the 3-qubit basis (q0, q1, q2).
# Ideal Decoded state: |000>
# Since the state is 3-qubit (8 elements):
ideal_final_data_state = Statevector([1, 0, 0, 0, 0, 0, 0, 0])

F_no_error_detected = np.abs(ideal_final_data_state.inner(final_decoded_state))**2

print(f"Encoded Fidelity (Syndrome 00 Detected):")
print(f"Probability of '00' syndrome (P_00): {P_00:.8f}")
print(f"Calculated Fidelity (F_no error detected): {F_no_error_detected:.8f}")
print(f"Approximation (1 - epsilon**6): {1 - epsilon**6:.8f}\n")

# --- 4. Conclusion and Comparison ---

if F_no_error_detected > Funencoded:
    print("Conclusion: The fidelity of the encoded, corrected state (F_no_error_detected) is significantly higher")
    print("than the unencoded state fidelity (F_unencoded), demonstrating the suppression of error probability.")
    print(f"The error is suppressed from O(epsilon^2) to O(epsilon^6).")
else:
    print("Conclusion: The fidelity check did not show the expected suppression. Review the circuit implementation.")

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace, state_fidelity, DensityMatrix, Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetDensityMatrix
import matplotlib.pyplot as plt

results_dir = 'results/BBPSSW/'
import os
os.makedirs(results_dir, exist_ok=True)

def get_werner_state(F):
    """
    Constructs a Werner State (Density Matrix) with exact Fidelity F.
    """
    # Create Bell Basis
    phi_plus = (Statevector.from_label('00') + Statevector.from_label('11')) / np.sqrt(2)
    phi_minus = (Statevector.from_label('00') - Statevector.from_label('11')) / np.sqrt(2)
    psi_plus = (Statevector.from_label('01') + Statevector.from_label('10')) / np.sqrt(2)
    psi_minus = (Statevector.from_label('01') - Statevector.from_label('10')) / np.sqrt(2)

    # Convert to Density Matrices
    rho_phi_plus = DensityMatrix(phi_plus)
    rho_others = (DensityMatrix(phi_minus) + 
                  DensityMatrix(psi_plus) + 
                  DensityMatrix(psi_minus))

    # Create Mixture
    rho_werner = F * rho_phi_plus + ((1 - F) / 3) * rho_others
    return rho_werner

def get_ideal_bell_state():
    psi = (Statevector.from_label('00') + Statevector.from_label('11')) / np.sqrt(2)
    return DensityMatrix(psi)

def run_werner_experiment(input_fidelity=0.8):
    # --- 1. Create the Werner State for ONE pair ---
    rho_pair = get_werner_state(input_fidelity)
    
    # --- 2. Create the Full 4-Qubit State (Tensor Product) ---
    # We combine the state of Pair 2 (Q2,Q3) and Pair 1 (Q0,Q1)
    # rho_total becomes a 16x16 matrix
    rho_total = rho_pair.tensor(rho_pair)
    
    qc = QuantumCircuit(4)

    # --- 3. Initialize ALL qubits at once ---
    # This fixes the "Incorrect Shape" error.
    qc.append(SetDensityMatrix(rho_total), [0, 1, 2, 3])

    # Save state before protocol
    qc.save_density_matrix(label='state_before')

    # --- 4. Run BBPSSW Protocol ---
    # Alice CNOT (0 -> 2)
    qc.cx(0, 2)
    # Bob CNOT (1 -> 3)
    qc.cx(1, 3)

    # Save state after protocol
    qc.save_density_matrix(label='state_after')

    # --- 5. Simulation ---
    sim = AerSimulator(method='density_matrix')
    qc_transpiled = transpile(qc, sim)
    result = sim.run(qc_transpiled).result()
    # print(qc.draw('text'))
    print(qc.draw('mpl', filename=results_dir+'PurificationCircuit.png', fold=0))
    
    return result

def calculate_fidelities(result):
    data = result.data()
    rho_after_full = data['state_after']
    ideal_bell = get_ideal_bell_state()

    # --- Post-Selection Logic ---
    op_0 = Operator.from_label('0')
    op_1 = Operator.from_label('1')
    op_I = Operator.from_label('I')

    # Projectors for Q3, Q2, Q1, Q0
    P00 = op_0.tensor(op_0).tensor(op_I).tensor(op_I)
    P11 = op_1.tensor(op_1).tensor(op_I).tensor(op_I)

    # Apply projection
    rho_match_00 = rho_after_full.evolve(P00)
    rho_match_11 = rho_after_full.evolve(P11)
    
    rho_post_selected = rho_match_00 + rho_match_11
    prob_success = np.real(rho_post_selected.trace())
    
    if prob_success > 0:
        rho_post_selected = rho_post_selected / prob_success
    
    # Trace out sacrificial qubits (2 and 3)
    rho_pair1_final = partial_trace(rho_post_selected, [2, 3])
    
    f_out = state_fidelity(rho_pair1_final, ideal_bell)

    return f_out, prob_success

if __name__ == "__main__":

    fin_values = np.linspace(0.5, 1.0, 11)
    fout = []
    success_probs = []
    for F_in in fin_values:
        print("=" * 50)
        # print("-" * 50)
        print(f"Running BBPSSW with Werner State (F_in = {F_in})...")
        
        try:
            res = run_werner_experiment(input_fidelity=F_in)
            f_out, success = calculate_fidelities(res)
            
            print("-" * 30)
            print(f"Input Fidelity  : {F_in:.4f}")
            print(f"Final Fidelity  : {f_out:.4f}")
            # Print Theoretical Final Fidelity
            F_theory = (F_in**2 + (1 - F_in)**2 / 9) / (F_in**2 + (2 * F_in * (1 - F_in)) / 3 + (5 * (1 - F_in)**2) / 9)
            print(f"Theoretical F_out: {F_theory:.4f}")
            print(f"Success Probability: {success:.2%}")
            print("-" * 30)
            fout.append(f_out)
            success_probs.append(success)
            if f_out > F_in:
                print("SUCCESS: Fidelity improved!")
            else:
                print("NOTE: No improvement.")
                
        except Exception as e:
            print(f"Error: {e}")

    # Optionally, plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(fin_values, fout, marker='o', label='Final Fidelity')
    plt.plot(fin_values, fin_values, linestyle='--', label='Input Fidelity (Reference)')
    plt.plot(fin_values, np.array(success_probs), marker='x', label='Success Probability')
    plt.xlabel('Input Fidelity (F_in)')
    plt.ylabel('Fidelity')
    plt.title('BBPSSW Protocol: Input vs Final Fidelity')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir + 'FidelityPlot.png')
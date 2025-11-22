import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace, state_fidelity, DensityMatrix, Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetDensityMatrix
import matplotlib.pyplot as plt

results_dir = 'results/DEJMPS/'
import os
os.makedirs(results_dir, exist_ok=True)

def get_bell_diagonal_state(p_00, p_01, p_10, p_11):
    """
    Creates a Bell Diagonal Density Matrix.
    p_00: Phi+ (Ideal), p_01: Phi- (Phase Error), p_10: Psi+ (Bit Error), p_11: Psi- (Bit+Phase)
    """
    phi_plus = Statevector.from_label('00') + Statevector.from_label('11')
    phi_minus = Statevector.from_label('00') - Statevector.from_label('11')
    psi_plus = Statevector.from_label('01') + Statevector.from_label('10')
    psi_minus = Statevector.from_label('01') - Statevector.from_label('10')

    # Normalize
    phi_plus /= np.linalg.norm(phi_plus)
    phi_minus /= np.linalg.norm(phi_minus)
    psi_plus /= np.linalg.norm(psi_plus)
    psi_minus /= np.linalg.norm(psi_minus)

    rho = (p_00 * DensityMatrix(phi_plus) +
           p_01 * DensityMatrix(phi_minus) +
           p_10 * DensityMatrix(psi_plus) +
           p_11 * DensityMatrix(psi_minus))
    return rho

def get_ideal_bell_state():
    psi = (Statevector.from_label('00') + Statevector.from_label('11')) / np.sqrt(2)
    return DensityMatrix(psi)

def run_purification(protocol_type, input_state):
    qc = QuantumCircuit(4)
    
    # Initialize 4 qubits (2 copies of input state)
    rho_total = input_state.tensor(input_state)
    qc.append(SetDensityMatrix(rho_total), [0, 1, 2, 3])

    # --- Protocol Specifics ---
    if protocol_type == 'DEJMPS':
        # CORRECTED ROTATION:
        # Alice (0, 2) rotates by +pi/2
        # Bob (1, 3) rotates by -pi/2 (Conjugate)
        # This "twirl" converts Phase errors (Phi-) into Bit errors (Psi+)
        qc.rx(np.pi/2, [0, 2]) 
        qc.rx(-np.pi/2, [1, 3]) 
    
    # --- Bilateral CNOTs (Common to both) ---
    # Alice: CNOT 0 -> 2
    qc.cx(0, 2)
    # Bob: CNOT 1 -> 3
    qc.cx(1, 3)

    qc.save_density_matrix(label='final_rho')

    # Run simulation
    sim = AerSimulator(method='density_matrix')
    qc = transpile(qc, sim)
    result = sim.run(qc).result()
    rho_final_full = result.data()['final_rho']
    if protocol_type == 'DEJMPS':
        print(qc.draw('mpl', filename=results_dir+'PurificationCircuit.png', fold=0))
    
    # --- Post-Selection Logic ---
    # Measure Target Pair (Q2, Q3). Keep if 00 or 11.
    op_0 = Operator.from_label('0')
    op_1 = Operator.from_label('1')
    op_I = Operator.from_label('I')

    # Projectors (Order: Q3, Q2, Q1, Q0)
    P00 = op_0.tensor(op_0).tensor(op_I).tensor(op_I)
    P11 = op_1.tensor(op_1).tensor(op_I).tensor(op_I)

    rho_success = rho_final_full.evolve(P00) + rho_final_full.evolve(P11)
    
    # Normalize
    prob_success = np.real(rho_success.trace())
    if prob_success > 0:
        rho_success = rho_success / prob_success
        
    # Trace out sacrificed pair (Q2, Q3) to get purified pair (Q0, Q1)
    rho_purified = partial_trace(rho_success, [2, 3])
    
    return rho_purified, prob_success

if __name__ == "__main__":
    # Case: Input with heavy Phase Noise (Phi-)
    # BBPSSW is bad at correcting Phase noise directly.

    fin_values = np.linspace(0.5, 1.0, 11)
    fout_bbpssw = []
    fout_dejmps = []
    # success_probs = []
    # success_probs = []
    for F in fin_values:
        print("=" * 50)
        # F = 0.8
        # State: 80% Phi+, 15% Phi- (Phase Error), 2.5% X, 2.5% Y
        p_01 = 0.75 * (1 - F)
        p_10 = p_01 / 6
        p_11 = p_10
        input_rho = get_bell_diagonal_state(p_00=F, p_01=p_01, p_10=p_10, p_11=p_11)
        
        ideal_state = get_ideal_bell_state()
        start_fidelity = state_fidelity(input_rho, ideal_state)
        
        print(f"--- Experiment Setup ---")
        print(f"Initial Fidelity: {start_fidelity:.4f}")
        print(f"Noise Profile   : Biased (Heavy Z-error / Phi-)")

        # 1. Run BBPSSW
        rho_bbpssw, prob_b = run_purification('BBPSSW', input_rho)
        fid_bbpssw = state_fidelity(rho_bbpssw, ideal_state)
        fout_bbpssw.append(fid_bbpssw)
        
        # 2. Run DEJMPS
        rho_dejmps, prob_d = run_purification('DEJMPS', input_rho)
        fid_dejmps = state_fidelity(rho_dejmps, ideal_state)
        fout_dejmps.append(fid_dejmps)  

        print(f"\n--- Results ---")
        print(f"BBPSSW Output Fidelity: {fid_bbpssw:.4f}")
        print(f"DEJMPS Output Fidelity: {fid_dejmps:.4f}")
        
        print("\n--- Analysis ---")
        if fid_bbpssw < start_fidelity:
            print("BBPSSW FAILED to purify. Fidelity dropped.")
        else:
            print("BBPSSW SUCCEEDED.")
        
        if fid_dejmps >= start_fidelity:
            print("DEJMPS SUCCEEDED.")
        else:
            print("DEJMPS FAILED to purify. Fidelity dropped.")

    # Optionally, plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(fin_values, fin_values, linestyle='--', label='Input Fidelity (Reference)')
    plt.scatter(fin_values, fout_bbpssw, marker='o', label='BBPSSW Final Fidelity', zorder = 3, color='green')
    plt.scatter(fin_values, fout_dejmps, marker='o', label='DEJMPS Final Fidelity', zorder = 3, color='red')
    plt.xlabel('Input Fidelity (F_in)')
    plt.ylabel('Fidelity')
    plt.title('BBPSSW Protocol: Input vs Final Fidelity')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir + 'FidelityPlot.png')
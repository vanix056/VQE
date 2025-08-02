from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.converters.second_quantization import JordanWignerMapper

def create_deuteron_hamiltonian(N, hbar_omega=7.0, V0=-5.68658111):
    hamiltonian_terms = {}
    for m in range(N):
        for n in range(N):
            # Label for FermionicOp: creation at m, annihilation at n
            label = f"+_{m} -_{n}"
            # Kinetic term <m|T|n>
            kinetic = (hbar_omega/2) * (
                (2*m + 1.5)*int(m==n)
                - np.sqrt(m*(m+0.5))*int(m==n+1)
                - np.sqrt((m+1)*(m+1.5))*int(m==n-1)
            )
            # Potential term <m|V|n>
            potential = V0 * int(m==0 and n==0) * int(m==n)
            hamiltonian_terms[label] = kinetic + potential
    # Build FermionicOp
    fermion_op = FermionicOp(hamiltonian_terms, num_spin_orbitals=N)
    # Map to qubits
    mapper = JordanWignerMapper()
    qubit_ham = mapper.map(fermion_op)
    return qubit_ham

# Build qubit Hamiltonians H1, H2, H3...
qubit_hamiltonians = [create_deuteron_hamiltonian(N) for N in range(1,4)]

# Set up VQE for each Hamiltonian
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP

for i,H in enumerate(qubit_hamiltonians, start=1):
    num_qubits = H.num_qubits
    ansatz = TwoLocal(num_qubits, ['rz','ry'], 'cx', reps=i)
    vqe = VQE(ansatz=ansatz, optimizer=SLSQP())
    result = vqe.compute_minimum_eigenvalue(H)
    print(f"H_{i} binding energy ≈ {result.optimal_value} MeV")

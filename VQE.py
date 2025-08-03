import numpy as np
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import QuantumInstance
from qiskit import Aer

def create_deuteron_hamiltonian(N, hbar_omega=7.0, V0=-5.68658111):
    terms = []
    for m in range(N):
        for n in range(N):
            term_str = f"+_{m} -_{n}"
            kinetic = (hbar_omega/2) * (
                (2*m + 1.5)*int(m==n)
                - np.sqrt(m*(m+0.5))*int(m==n+1)
                - np.sqrt((m+1)*(m+1.5))*int(m==n-1)
            )
            potential = V0 * int(m==n==0)
            coefficient = kinetic + potential
            
            if not np.isclose(coefficient, 0):
                terms.append((term_str, coefficient))
    
    fermion_op = FermionicOp(terms, register_length=N)
    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper)
    qubit_ham = converter.convert(fermion_op)
    return qubit_ham

# Create quantum backend
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)

qubit_hamiltonians = [create_deuteron_hamiltonian(N) for N in range(1,4)]

for i, H in enumerate(qubit_hamiltonians, start=1):
    num_qubits = H.num_qubits
    ansatz = TwoLocal(num_qubits, ['rz','ry'], 'cx', reps=i)
    vqe = VQE(ansatz=ansatz, optimizer=SLSQP(), quantum_instance=quantum_instance)
    result = vqe.compute_minimum_eigenvalue(H)
    print(f"H_{i} binding energy ≈ {result.optimal_value} MeV")
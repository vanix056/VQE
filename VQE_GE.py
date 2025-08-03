from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP
from qiskit.opflow import PauliSumOp, PauliExpectation
from qiskit.utils import QuantumInstance

# Convert your SparsePauliOp to PauliSumOp (opflow format)
qubit_hamiltonians = [
    PauliSumOp.from_list([("I", 1.0)]),
    PauliSumOp.from_list([("Z", 1.0)]),
    PauliSumOp.from_list([("ZZ", 1.0)])  # Example for H_3
]

H3 = qubit_hamiltonians[2]
num_qubits = H3.num_qubits

# Define ansatz
ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cx', reps=2, entanglement='full')

# Set up QuantumInstance
quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

# Set up VQE with PauliExpectation
vqe = VQE(ansatz=ansatz, 
          optimizer=SLSQP(), 
          quantum_instance=quantum_instance,
          expectation=PauliExpectation())

# Run VQE
vqe_result = vqe.compute_minimum_eigenvalue(H3)
print("VQE ground state energy:", vqe_result.optimal_value)
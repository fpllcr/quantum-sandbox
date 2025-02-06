import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
import matplotlib.pyplot as plt


# ==============
# define your layer
# ==============


def layer(x, num_qubits):
    """
    Constructs a quantum circuit using basis embedding and a sequence of Hadamard and CNOT gates.

    This circuit is designed to generate a pattern of quantum operations on a set of qubits.
    The function does not have an explicit `return` statement because its purpose is to modify the
    quantum state within the context of PennyLane. Functions of this type add quantum operations to a circuit,
    instead of returning a traditional value.

    Args:
        x (array-like): Input data to be encoded into the qubits.
        num_qubits (int): The number of qubits to be used in the circuit.

    Operations performed:
        1. Applies `qml.AngleEmbedding(x, wires, rotation)` to encode the input data into the quantum state.
            qml.AngleEmbedding(x) takes the ca

        2. Applies a Hadamard gate to each qubit.
        3. Applies CNOT gates between adjacent qubits and connects the last qubit to the first one,
           creating a circular CNOT structure.

    The function does not return any values, but rather constructs the quantum circuit within the PennyLane environment.
    """

    wires = range(num_qubits)
    qml.AngleEmbedding(x, wires=range(num_qubits), rotation="X")
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        if j != num_qubits - 1:
            qml.CNOT(wires=[j, j + 1])
        else:
            qml.CNOT(wires=[j, 0])


def ansatz(x, num_qubits):
    """
    Applies the ansatz for a quantum circuit by invoking the `layer` function.

    Args:
        x (array-like): Input parameters to be embedded into the quantum circuit.
        wires (Iterable[int]): The qubits (wires) on which the circuit will act.

    The function acts as a wrapper for the `layer(x)` function, applying the layer of quantum gates to
    the specified qubits. It is intended for use as part of a variational quantum algorithm (VQA)
    or quantum machine learning task.

    Note:
        This function does not return any values, as it directly modifies the quantum state
        through the operations defined in the `layer(x)` function.
    """
    layer(x, num_qubits)


# ========
# quantum kernel
# ========


def q_kernel(num_qubits, ansatz_func):
    """
    Constructs a quantum kernel using a provided ansatz and builds the corresponding
    kernel circuit function.

    Args:
        num_qubits (int): Number of qubits to use in the quantum circuit.
        ansatz_func (function): The ansatz function defining the quantum circuit.

    Returns:
        kernel_function (function): A function that can be used as a kernel in SVC.
    """
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)
    wires = dev.wires.tolist()

    adjoint_ansatz = qml.adjoint(ansatz_func)

    @qml.qnode(dev, interface="autograd")
    def kernel_circuit(x1, x2):
        ansatz_func(x1, num_qubits)
        adjoint_ansatz(x2, num_qubits)
        return qml.probs(wires=range(num_qubits))

    def kernel(x1, x2):
        return kernel_circuit(x1, x2)[0]

    return kernel


# =======================
# dictionary, different kernels: classical and quantum
# =======================
"""We generate a DICTIONARY FOR CHOOSING A KERNEL 

Notice that the standard way of svc is
clf = SVC(kernel="poly", degree=2, coef0=0)

for chosing between different Kernels, i define the  dictionary
"""


svm_models = {
    "linear": {
        "model": SVC(kernel="linear", C=1.0),
        "kernel_matrix": lambda X: linear_kernel(X, X),  # Use sklearn linear kernel
    },
    "poly": {
        "model": SVC(kernel="poly", degree=2, coef0=0, C=1.0),
        "kernel_matrix": lambda X: polynomial_kernel(
            X, X, degree=2, coef0=0
        ),  # Polynomial kernel
    },
    "rbf": {
        "model": SVC(kernel="rbf", gamma="scale", C=1.0),
        "kernel_matrix": lambda X: rbf_kernel(X, X),  # RBF kernel
    },
    "sigmoid": {
        "model": SVC(kernel="sigmoid", coef0=0, C=1.0),
        "kernel_matrix": lambda X: sigmoid_kernel(X, X),  # Sigmoid kernel
    },
    "quantum": {
        "model": lambda num_qubits: SVC(
            kernel=lambda X1, X2: qml.kernels.kernel_matrix(
                X1, X2, q_kernel(num_qubits, ansatz)
            )
        ),
        "kernel_matrix": lambda X, num_qubits: qml.kernels.kernel_matrix(
            X, X, q_kernel(num_qubits, ansatz)
        ),
    },
}

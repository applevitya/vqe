# optimization by Adam algorithm
import pennylane as qml
import numpy as np
from math import *
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dev = qml.device('default.qubit', wires=2)

def U(phi, delta):
    T = cos(delta / 2) + 1j * sin(delta / 2) * cos(2 * phi)
    R = 1j * sin(delta / 2) * sin(2 * phi)
    return np.array([[T, R], [-R.conjugate(), T.conjugate()]])

@qml.qnode(dev, interface='tf')
def my_sheme(phi):
    qml.QubitUnitary(U(phi[0],pi/2), wires = 0)
    qml.PauliX(wires = 1)
    qml.QubitUnitary(U(phi[1],pi),wires=0)
    qml.CNOT(wires=[0,1])
    qml.QubitUnitary(U(phi[2],pi/2),wires=0)
    qml.QubitUnitary(U(phi[4],pi/2),wires=1)
    qml.QubitUnitary(U(phi[3],pi),wires=0)
    qml.QubitUnitary(U(phi[5],pi),wires=1)


    return qml.state()#qml.expval(qml.PauliZ(0))

print(my_sheme([1,1,1,1,1,1]))


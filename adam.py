# optimization by Adam algorithm
import pennylane as qml
import numpy as np
from math import *
import tensorflow as tf
import tensorflow_probability as tfp
import os
import torch
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate, YGate, ZGate, CCXGate



state_zero = np.array([[1.0], [0.0]]);
I = IGate().to_matrix();
X = XGate().to_matrix();
Y = YGate().to_matrix();
Z = ZGate().to_matrix();
#####################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dev = qml.device('default.qubit', wires=2)

def U(phi, delta):
    T = cos(delta / 2) + 1j * sin(delta / 2) * cos(2 * phi)
    R = 1j * sin(delta / 2) * sin(2 * phi)
    return np.array([[T, R], [-R.conjugate(), T.conjugate()]])
def schwinger_matrix(k):
    return {
        k == 1: np.kron(I,I), # II
        k == 2: np.kron(X,X), # XX
        k == 3: np.kron(Y,Y), # YY
        k == 4: np.kron(Z,Z), # ZZ
        k == 5: np.kron(Z,I),  # ZI
        k == 6: np.kron(I,Z)   # IZ
    }[True]

#@qml.qnode(dev, diff_method="finite-diff",interface='tf', mutable = True)
def my_sheme(phi,wires):
    qml.RY(2*phi[0],wires=0).adjoint()
    qml.RZ(pi/2,wires = 0).adjoint()
    qml.RY(2 * phi[0],wires= 0)
    qml.PauliX(wires = 1)

    qml.RY(2 * phi[1],wires= 0).adjoint()
    qml.RZ(pi,wires = 0).adjoint()
    qml.RY(2 * phi[1],wires= 0)
    qml.CNOT(wires=[0,1])

    qml.RY(2 * phi[2],wires=0).adjoint()
    qml.RZ(pi/2,wires= 0).adjoint()
    qml.RY(2 * phi[2],wires= 0 )

    qml.RY(2 * phi[4],wires=1).adjoint()
    qml.RZ(pi/2,wires=1).adjoint()
    qml.RY(2 * phi[4],wires=1)

    qml.RY(2 * phi[3],wires=0).adjoint()
    qml.RZ(pi,wires=0).adjoint()
    qml.RY(2 * phi[3],wires=0)

    qml.RY(2 * phi[5],wires=1).adjoint()
    qml.RZ(pi,wires = 1).adjoint()
    qml.RY(2 * phi[5],wires=1)

    #return qml.state()#qml.expval(qml.PauliZ(0))


schwinger = schwinger_matrix(1) + schwinger_matrix(2)+schwinger_matrix(3)+0.5*(-schwinger_matrix(5)+schwinger_matrix(4)+0*schwinger_matrix(6)-0*schwinger_matrix(5))

obs = qml.Hermitian(schwinger, wires=[0, 1])
H = qml.Hamiltonian((1.0,), (obs,))



cost = qml.ExpvalCost(my_sheme, H, dev,interface="tf",diff_method = 'best')


phi = tf.Variable([1.0,1.0,0.0,0.0,0.0,0.0])
print(cost(phi))

loss_fn = lambda: cost(phi)


losses = tfp.math.minimize(loss_fn,trainable_variables=[phi],
                           num_steps=60,
                           optimizer=tf.optimizers.Adam(learning_rate=0.1))

print(losses.numpy())



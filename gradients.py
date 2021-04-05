from qiskit import *
from math import *
from numpy.random import random, multinomial
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate, YGate, ZGate, CCXGate
import numpy as np
from computings import C_Gate
####################################################################################

I = IGate().to_matrix();
X = XGate().to_matrix();
Y = YGate().to_matrix();
Z = ZGate().to_matrix();
CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]);
state_zero = np.array([[1.0], [0.0]]);

psi0 = np.kron(np.kron(state_zero,state_zero),state_zero) # |000>
psi1 = np.kron(HGate().to_matrix(),np.kron(I,I))          # HII H - hadamard Gate


####################################################################################
def schwinger_matrix(k):
    return {
        k == 1: np.kron(I,I), # II
        k == 2: np.kron(X,X), # XX
        k == 3: np.kron(Y,Y), # YY 
        k == 4: np.kron(Z,Z), # ZZ
        k == 5: np.kron(Z,I),  # ZI
        k == 6: np.kron(I,Z)   # IZ
    }[True]

def U(phi, delta):
    """Return waveplate matrix with retardance delta and axis angle phi.

    delta = pi for HWP
    delta = pi/2 for QWP
    """
    T = cos(delta / 2) + 1j * sin(delta / 2) * cos(2 * phi)
    R = 1j * sin(delta / 2) * sin(2 * phi)
    return np.array([[T, R], [-R.conjugate(), T.conjugate()]]) 

def d_U(phi, delta,i): # derivative of U_operator  i - circuit B (B1 or B2) 
    Y = RYGate(2 * phi).to_matrix()
    Z = RZGate(delta).to_matrix()
    return {
        i == 1: RYGate(2 * phi + pi).to_matrix()@Z.conjugate()@Y.transpose(),
        i == 2: -Y@Z.conjugate()@RYGate(-2*phi+pi).to_matrix()
    }[True] 

def U_circuit(phi, N,i): # N = derivative angle number   i - circuit B (B1 or B2) 
    u = {1: U(phi[0],pi/2), 2: U(phi[1],pi), 3: U(phi[2],pi/2), 4: U(phi[3],pi), 5: U(phi[4],pi/2), 6: U(phi[5],pi)}
    if N==0:
        pass
    else:
        u[N] = d_U(phi[N-1],pi*(0.5 + 0.5*((N+1)%2)),i)
    return  np.kron(u[4]@u[3],u[6]@u[5])@CX@np.kron(u[2]@u[1],X)



def B(phi,N,k,i): # i - circuit B (B1 or B2) 
    return (U_circuit(phi,0,i).conjugate().transpose())@schwinger_matrix(k)@U_circuit(phi,N,i)

def hadamard(phi,N,k,n):
    cir = psi1@C_Gate(B(phi,N,k,1),3)@psi1@psi0
    cir2  = psi1@C_Gate(B(phi,N,k,2),3)@psi1@psi0
    if type(n) == str:
        t = np.sum((np.abs(cir)**2)[0:4])
        t2 = np.sum((np.abs(cir2)**2)[0:4])
        return 4*t+4*t2-4
    else:
        d = [np.abs(amp[0])**2 for amp in cir]
        d2 = [np.abs(amp[0])**2 for amp in cir2]
        counts = np.random.multinomial(n, d)
        counts2 = np.random.multinomial(n,d2)
        t = np.sum(counts[0:4])/n   # вероятность найти 1 кубит в |0>
        t2 = np.sum(counts2[0:4])/n
        return  4*t+4*t2-4



def hadamard_test(phi,N,m,n): #II+XX+YY+0.5(-ZI+ZZ+mIZ-mZI)
    return hadamard(phi,N,1,n)+hadamard(phi,N,2,n)+hadamard(phi,N,3,n)+0.5*hadamard(phi,N,4,n)-0.5*hadamard(phi,N,5,n)+0.5*m*hadamard(phi,N,6,n)-0.5*m*hadamard(phi,N,5,n)

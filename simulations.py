import numpy as np
from math import *
from qiskit.extensions import RXGate, RZGate, RYGate, HGate, XGate, IGate, CXGate, YGate, ZGate, CCXGate
from gradients import hadamard_test,U_circuit,schwinger_matrix
from scipy.optimize import minimize
########################################################

state_zero = np.array([[1.0], [0.0]]);

######################################################
def energy_schwinger(phi,m,n):
    psi = U_circuit(phi,0,1) @ np.kron(state_zero,state_zero)
    if type(n) == str:
        schwinger = schwinger_matrix(1) + schwinger_matrix(2)+schwinger_matrix(3)+0.5*(-schwinger_matrix(5)+schwinger_matrix(4)+m*schwinger_matrix(6)-m*schwinger_matrix(5))
        return (psi.transpose().conjugate()@schwinger@psi)[0][0].real

    else:
        H = np.array([[1.0], [0.0]])
        V = np.array([[0.0], [1.0]])
        D = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        A = np.array([[1 / np.sqrt(2)], [-1 / np.sqrt(2)]])
        R = np.array([[1 / np.sqrt(2)], [1j / np.sqrt(2)]])
        L = np.array([[1 / np.sqrt(2)], [-1j / np.sqrt(2)]])

        def p(basis1,basis2):#probabilities
            M = np.dot((np.kron(basis1, basis2)),(np.transpose(np.kron(basis1, basis2).conj())))
            return np.trace(np.dot(M, np.dot(psi, np.transpose(psi.conj())))).real 

        p_HV = np.random.multinomial(n, [p(H, H), p(H, V), p(V, H),p(V, V)], 1)
        p_DA = np.random.multinomial(n, [p(D, D), p(D, A), p(A, D),p(A, A)], 1)
        p_RL = np.random.multinomial(n, [p(R, R), p(R, L), p(L, R),p(L, L)], 1)

        II = p_HV[0][0] + p_HV[0][1] + p_HV[0][2] + p_HV[0][3]
        XX = p_DA[0][0] - p_DA[0][1] - p_DA[0][2] + p_DA[0][3]
        YY = p_RL[0][0] - p_RL[0][1] - p_RL[0][2] + p_RL[0][3]
        ZZ = p_HV[0][0] - p_HV[0][1] - p_HV[0][2] + p_HV[0][3]
        IZ = p_HV[0][0] - p_HV[0][1] + p_HV[0][2] - p_HV[0][3]
        ZI = p_HV[0][0] + p_HV[0][1] - p_HV[0][2] - p_HV[0][3]

        return (II+XX+YY+0.5*(ZZ-ZI + m*IZ - m*ZI))/n #1+XX+YY+0.5(-ZI+ZZ+mIZ-mZI)




############# OPTIMIZATION ####################################

x0 = np.random.uniform(0, 2 * pi, 6)

def optimization(x0):
    points = []
    m = 0
    n1 = 1000 
    def callback_func(x):
        points.append(energy_schwinger(x,m,'100000'))
        return False

    def target_func(x):
        return energy_schwinger(x,m,'100000')

    def gradients(x0):
        der = np.zeros_like(x0)
        for i in range(0, len(x0)):
            j = i + 1
            der[i] = hadamard_test(x0,j,m,'n1')
        return der


    result = minimize(target_func, x0=x0, callback=callback_func, method="SLSQP",jac = gradients,options={'disp':True, 'maxiter': 200, 'eps': 0, "ftol":0})


optimization(x0)


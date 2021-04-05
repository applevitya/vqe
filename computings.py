import numpy as np
import math as math

# generalized Controlled Gate
def C_Gate(U, q): # U - numpy array; n - number of qibits
    d = int(math.pow(2,q)) # dimension
    C = [[0 for j in range(d)] for i in range(d)]
    C = np.empty((d,d), dtype="object")
    for i in range(0, d):
        for j in range(0, d):
            if i == j:
                C[i][j] = 1
            else:
                C[i][j] = 0

    for m in range(int(d/2),d):
        for n in range(int(d/2),d):
            C[n][m] = U[n-int(d/2)][m-int(d/2)]
    return C




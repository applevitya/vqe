import numpy as np
from scipy.optimize import OptimizeResult

def minimize_spsa(func, x0, args=(), callback=None, maxiter=200,
                  ea=0.602, a0=1.0, af=0.0,
                  eb=0.101, b0=1.0, bf=0.0):
    """Minimize specified function using SPSA algorithm.
    
    Parameters
    ----------
    func: callable
        Function under minimization.
        Internally is called as `func(x, *args)`.
    
    x0: array-like
        Initial guess.
        
    args: tuple
        Extra arguments passed to function.
        
    maxiter: int
        Maximal number of iterations.
        
    ea, a0, af: float
        Parameters for step size scaling (exponent, initial, final).
        Step size `a` at iteration `k` is:
        `a = (a0-af)/(k+1)**ea + af`.
        
    eb, b0, bf: float
        Parameters for step size scaling during gradient estimation
        (exponent, initial, final).
        Step size `b` at iteration `k` is:
        `b = (b0-bf)/(k+1)**eb + bf`.
        
    callback: callable
        Called after each iteration as `callback(xk)`, where `xk` are
        current parameters.
    """
    dim = len(x0)
    x = np.array(x0, dtype='float64')
    for k in range(maxiter):
        a = (a0-af)/(k+1)**ea + af
        b = (b0-bf)/(k+1)**eb + bf
        delta = np.random.choice([-1,1], size=dim)
        grad = (func(x + b*delta, *args) - func(x - b*delta, *args)) / (2*b*delta)
        x -= a*grad
        if callback:
            callback(x)
    msg = 'terminated after reaching max number of iterations'
    return OptimizeResult(fun=func(x), x=x, nit=maxiter, nfev=2*maxiter, message=msg, success=True)

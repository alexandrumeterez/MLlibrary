import numpy as np
import gradient_descent as grad
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs

from scipy.optimize import minimize_scalar

def BFGS(f, x0, grad_f, eps = 0.01, max_iterations = 100):
    H = np.identity(grad_f(*x0).shape[0])
    x = x0
    k = 1
    while k <= max_iterations:
        if np.linalg.norm(grad_f(*x)[0]) < eps:
            print("Minimized {0} after {1} iterations(optimal)".format(x, k))
            return x
        p = -np.dot(H, grad_f(*x0))
        
        h = lambda alpha: f(*(x + np.multiply(alpha, p)))
        #grad_h = lambda alpha: np.dot(grad_f(*(x + np.multiply(alpha, p))), p)
        optimized_alpha = minimize_scalar(h, method = 'brent')
        s = np.multiply(optimized_alpha.x, p)
        x = x + s
        y = grad_f(*(x + s)) - grad_f(*x)
        #rho = 1/np.dot(y, s)
        
        yk_sk = np.dot(y, s)
        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / yk_sk
        except ZeroDivisionError:
            rhok = 1000.0
            print("Divide-by-zero encountered: rho assumed large")
            
        first_factor = np.identity(grad_f(*x0).shape[0]) - np.outer(s, y) / np.dot(y, s)
        second_factor = np.identity(grad_f(*x0).shape[0]) - np.outer(y, s) / np.dot(y, s)
        H = np.matmul(np.matmul(first_factor, H), second_factor)
        H = H + np.outer(s, s) / np.dot(y, s)
        
        k = k + 1
    print("Minimized {0} after {1} iterations(suboptimal)".format(x, k))
    return x, k



def f(x, y):
    return 3*(x**2) + 5 * (y**2) + 6 + 5 * x + 7 * y
def grad_f(x, y):
    return np.array([6 * x + 5, 10 * y + 7])

print(BFGS(f, np.array([1, 1]), grad_f, eps = 0.1, max_iterations = 15))

x0 = np.array([1, 1])
fmin_bfgs(f, x0)
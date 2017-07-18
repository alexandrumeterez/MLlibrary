import numpy as np

def conjugate_grad(A, b, x, max_iterations = 100, eps = 0.01):
    i = 0
    r = b - np.dot(A, x)
    d = r
    delta_new = np.dot(r, r)
    delta_zero = delta_new
    
    while i < max_iterations and delta_new > (eps**2) * delta_zero:
        q = np.dot(A, d)
        alpha = delta_new / np.dot(d, q)
        x = x + np.multiply(alpha, d)
        if i % 50 == 0:
            r = b - np.dot(A, x)
        else:
            r = r - np.multiply(alpha, q)
        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + np.multiply(beta, d)
        i = i + 1
        
    return x

A = np.array([[1, 2, 3], [-1, 0, 1], [3, 1, 3]])
b = np.array([-5, -3, -3])

x = conjugate_grad(A, b, np.array([1, 2, 3]))
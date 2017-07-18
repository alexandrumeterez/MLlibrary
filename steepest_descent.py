import numpy as np

def steepest_descent(A, b, x, max_iterations = 49, eps = 0.001):
    """
        Gives the solution to Ax = b via the steepest descent method.
    """
    i = 0
    r = b - np.dot(A, x)
    delta = np.dot(r, r)
    delta_zero = delta
    
    while i < max_iterations and delta > (eps**2) * delta_zero:
        q = np.dot(A, r)
        alpha = delta / np.dot(r, q)
        x = x + np.multiply(alpha, r)
        if i % 50 == 0:
            r = b - np.dot(A, x)
        else:
            r = r - np.multiply(alpha, q)
        delta = np.dot(r, r)
        i = i + 1
        
    return x
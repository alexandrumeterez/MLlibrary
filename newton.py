def newton(x_init, f_prime, f_double_prime, eps = 0.001, max_iterations = 100):
    """
        Takes in the first 2 derivatives of f and an initial guess of a root x_init and returns a root of the derivative in order to find a global(or local) minima via Newton's method.
    """    
    k = 1
    while k <= max_iterations:
        x_next = x_init - (f_prime(x_init)/f_double_prime(x_init))
        if(abs(f_prime(x_init)) < eps):
            return x_init
        else:
            k = k + 1
            x_init = x_next
    return False
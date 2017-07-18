def gradient_descent(f, grad_f, x_init, eps = 0.001, max_iterations = 1000, alpha = 0.001):
    """
        Returns the minimum of x through gradient descent. Works in any
        number of variables.
    """
    
    x = x_init
    k = 1
    f_next = f(x)
    while k <= max_iterations:
        f_eval = f_next
        
        if(isinstance(x, list)):
            x[:] = [var - alpha * grad_f(x) for var in x]
        else:
            x = x - alpha * grad_f(x)
        f_next = f(x)
        if abs(f_next - f_eval) < eps:
            return x
        k = k + 1
    return False


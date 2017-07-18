def golden_section_search(f, a, b, eps = 0.01):
    """
        Takes in a function f(under the assumption that it is unimodular and an interval [a, b] and returns the minima via the Golden Section Search method.
        
    """
    phi = 1.618033988
    
    #Initial bracketing
    xl = b - (b-a)/phi
    xu = a + (b-a)/phi
 
    while abs(b-a) > eps:
        if f(xl)<f(xu): #Removing left interval
            b = xu
            xu = xl
            xl = b - (b-a)/phi
        else: #Removing right interval
            a = xl
            xl = xu
            xu = a + (b-a)/phi
    return (xu+xl)/2

import numpy as np

class LinearRegressor(object):
    error = np.Infinity
    theta = np.empty(0)
    X_data = np.empty(0)
    y_data = np.empty(0)

    def __init__(self):
        pass
    
    def __compute_cost(self, X, y, theta):
        """
            Computes the cost function for gradient descent.
        """
        m = len(y)
        J = 0
        
        J = np.sum(np.square((np.dot(X, theta) - y))) / (2 * m)
        self.error = J
        self.X_data = X
        self.y_data = y
        
        return J
    
    #TODO
    #Add more iterative solvers(conjugate gradient, newton etc)
    def __gradient_descent(self, X, y, theta, alpha = 0.1, num_iters = 100):
        """
            Applies gradient descent to the initial theta and iterates until
            convergence
        """
        m = len(y)
        J_history = np.zeros(num_iters)
        
        for i in range(1, num_iters):
            temp = np.matmul((np.dot(X, theta) - y), X)
            theta = theta - np.multiply(alpha/m, temp)
            J_history[i] = LinearRegressor.__compute_cost(self, X, y, theta)
        
        self.theta = theta
        return theta
    
    def fit_to_data(self, X, y, theta, alpha = 0.1, num_iters = 100):
        self.theta = self.__gradient_descent(X, y, theta, alpha, num_iters)
    
    def fit_normal_equation(self, X, y):
        self.theta = np.dot(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
    
    def predict_data(self, X_prediction):
        return np.dot(self.theta, X_prediction)

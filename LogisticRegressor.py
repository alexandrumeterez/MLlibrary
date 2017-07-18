import numpy as np

class LogisticRegressor(object):
    weights = np.empty(0)
    features = np.empty(0)
    target = np.empty(0)
    
    def __init__(self):
        pass
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __compute_log_likelihood(self, features, target, weights):
        scores = np.dot(features, weights)
        log_likelihood = np.sum(target * scores - np.log(1 + np.exp(scores)))
        
        self.features = features
        self.target = target
        
        return log_likelihood
    
    def __compute_gradient(self, features, target, predictions):
        temp = target - predictions
        gradient = np.dot(features.T, temp)
        
        return gradient
    
    def fit_to_data(self, features, target, num_iterations = 100, learning_rate = 0.01, add_intercept = False):
        
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        
        
        weights = np.zeros(features.shape[1])
        
        for step in range(num_iterations):
            scores = np.dot(features, weights)
            predictions = self.__sigmoid(scores)
            
            weights += learning_rate * self.__compute_gradient(features, target, predictions)
            
            if step % 10000 == 0:
                print(self.__compute_log_likelihood(features, target, weights))
        
        self.weights = weights
        
        return weights
   

import matplotlib.pyplot as plt
np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

lr = LogisticRegressor()
lr.fit_to_data(simulated_separableish_features, simulated_labels, num_iterations=100000, learning_rate=5e-5, add_intercept=True)
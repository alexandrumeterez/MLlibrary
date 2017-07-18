from scipy.spatial import distance
class KNearestNeighbours(object):
    def euc_dist(self, x, y):
        return distance.euclidean(x, y)
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = self.euc_dist(row, self.X_train[0])
        best_index = 0
        
        for index in range(1, len(self.X_train)):
            curr_dist = self.euc_dist(row, self.X_train[index])
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_index = index
        
        return self.y_train[best_index]
    
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

classifier = KNearestNeighbours()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
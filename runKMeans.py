from use.accuracy import accuracy
from use.k_means import KMeansModel
from use.mnist import Mnist

data = Mnist()
X = data.getX()
Y = data.getY()

k_means = KMeansModel()
k_means.fit(X)
Y_pred = k_means.labels

print(accuracy(Y, Y_pred))

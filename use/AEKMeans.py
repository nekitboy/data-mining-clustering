from keras.models import load_model

from use.accuracy import accuracy
from use.mnist import Mnist
from use.k_means import KMeansModel
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))[:10000]
x_test = np.reshape(x_test,  (len(x_test),  28, 28, 1))[:1000]
Y = y_train[:10000]
X = x_train

encoder = load_model("encode_model_saved.h5")
encoded = encoder.predict(X, batch_size=10000)


X1 = []
for sample in encoded:
    X1.append(np.array(sample).flatten())

k_means = KMeansModel()
k_means.fit(X1)
Y_pred = k_means.labels

print(accuracy(Y, Y_pred))


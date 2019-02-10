from time import time

import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks, Sequential
from use.accuracy import accuracy
from keras.models import load_model
from use.mnist import Mnist
from use.k_means import KMeansModel
from sklearn.cluster import KMeans


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example

        model.add(ClusteringLayer(n_clusters=10))


    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


# Load Mnist data
data = Mnist()
X = data.getX()
Y = data.getY()

X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), 28, 28, 1))

# Encode X with autoencoder
encoder = load_model("encoder_model_saved.h5")
encoded = encoder.predict(X, batch_size=256)

# Fit encoded X with K_means
k_means = KMeansModel()

k_means.fit(encoded)
Y_pred = k_means.labels
y_pred_last = np.copy(Y_pred)

# Prepare DEC


clustering_layer = ClusteringLayer(10, name='clustering', weights=[k_means.cluster_centers])(encoder.output)
dec_model = Model(inputs=encoder.input, outputs=clustering_layer)
dec_model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
maxiter = 800
batch_size = 256
tol = 1e-3
update_interval = 140



loss = 0
index = 0
index_array = np.arange(X.shape[0])
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = dec_model.predict(X, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        y_pred = q.argmax(1)
    print(ite)

    idx = index_array[index * batch_size: min((index+1) * batch_size, X.shape[0])]
    loss = dec_model.train_on_batch(x=X[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0
    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]



print(accuracy(Y, y_pred))
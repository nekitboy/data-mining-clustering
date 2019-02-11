from keras import Sequential
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))


def create_deep_dense_ae():
    # Размерность кодированного представления
    encoding_dim = 10

    # Энкодер
    input_img = Input(shape=(28, 28, 1))
    flat_img = Flatten()(input_img)
    x = Dense(500, activation='relu')(flat_img)
    x = Dense(500, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='linear')(x)

    # Декодер
    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(2000, activation='relu')(input_encoded)
    x = Dense(500, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    flat_decoded = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(flat_decoded)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


encoder, decoder, autoencoder = create_deep_dense_ae()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

autoencoder.fit(x_train, x_train,
                epochs=40,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder.save("encoder_model_saved.h5")
decoder.save("decoder_model_saved.h5")

import matplotlib.pyplot as plt


def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])

    plt.figure(figsize=(2 * n, 2 * len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i * n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

n = 10

imgs = x_test[:n]
encoded_imgs = encoder.predict(imgs, batch_size=n)
encoded_imgs[0]


decoded_imgs = decoder.predict(encoded_imgs, batch_size=n)

plot_digits(imgs, decoded_imgs)

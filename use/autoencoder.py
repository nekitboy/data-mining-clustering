from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

from keras.datasets import mnist
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test,  (len(x_test),  28, 28, 1))


def create_deep_conv_ae():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (7, 7), activation='relu', padding='same')(x)

    # На этом моменте представление  (7, 7, 1) т.е. 49-размерное

    input_encoded = Input(shape=(7, 7, 1))
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

encoder, decoder, autoencoder = create_deep_conv_ae()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


encoder.save("encode_model_saved.h5")
decoder.save("decode_model_saved.h5")

# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def plot_digits(*args):
#     args = [x.squeeze() for x in args]
#     n = min([x.shape[0] for x in args])
#
#     plt.figure(figsize=(2 * n, 2 * len(args)))
#     for j in range(n):
#         for i in range(len(args)):
#             ax = plt.subplot(len(args), n, i * n + j + 1)
#             plt.imshow(args[i][j])
#             plt.gray()
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#
#     plt.show()
#
# n = 10
#
# imgs = x_test[:n]
# encoded_imgs = encoder.predict(imgs, batch_size=n)
# encoded_imgs[0]
#
#
# decoded_imgs = decoder.predict(encoded_imgs, batch_size=n)
#
# plot_digits(imgs, decoded_imgs)
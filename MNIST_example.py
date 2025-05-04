import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from dense import Dense
from activations import Tanh
from mse import MSE, MSE_prime
from network import train, predict


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# neural network
network = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

# train
train(network, MSE, MSE_prime, x_train, y_train, epochs=100, learning_rate=0.1)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))


fig, axes = plt.subplots(4, 5, figsize=(12, 8))
fig.suptitle("MNIST Predictions", fontsize=16)

for i, ax in enumerate(axes.flat):
    img = x_test[i].reshape(28, 28)
    true_label = np.argmax(y_test[i])
    pred_label = np.argmax(predict(network, x_test[i]))

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {pred_label} / True: {true_label}")
    ax.axis('off')

# Save the figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # To make room for suptitle
plt.savefig("mnist_predictions.png")
plt.show()
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Hyperparameters for Sparse SVD
fc_id = 49  # FC6 Layer Number
rank = 64
sr = 0.5  # Sparsity rate
rr = 0.5  # Reduced rank rate

def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

# Load CIFAR-10 data (using half of the dataset)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
half_train_size = x_train.shape[0] // 2
x_train, y_train = x_train[:half_train_size], y_train[:half_train_size]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train, x_test = normalize(x_train, x_test)

# Load the pre-trained model (ensure you have 'cifar10vgg.json' and 'cifar10vgg.h5' in the directory)
json_file = open('cifar10vgg.json', 'r')
cifar10_model_json = json_file.read()
json_file.close()
cifar10_model = model_from_json(cifar10_model_json)
cifar10_model.load_weights("cifar10vgg.weights.h5")
cifar10_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

def sparse_SVD_wr(weights, U, S, V, keep, sr, rr):
    tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]

    # Input node selection
    iwm = np.sum(abs(weights), axis=1)
    imid = sorted(iwm)[int(weights.shape[0] * sr)]
    ipl = np.where(iwm < imid)[0]

    # Output node selection
    owm = np.sum(abs(weights), axis=0)
    omid = sorted(owm)[int(weights.shape[1] * sr)]
    opl = np.where(owm < omid)[0]

    # Masking the weights
    subrank = int(keep * rr)
    for ind in ipl:
        tU[ind, subrank:] = 0

    for ind in opl:
        tV[subrank:, ind] = 0

    return tU, tS, tV

# FC6 Layer Weights
fc1 = cifar10_model.layers[fc_id].get_weights()
weights = fc1[0]

# Decompose weights and sparsify
U, S, V = np.linalg.svd(weights, full_matrices=False)
tU, tS, tV = sparse_SVD_wr(weights, U, S, V, rank, sr, rr)

weights_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
fc1[0] = weights_t
cifar10_model.layers[fc_id].set_weights(fc1)

# Evaluate the model
score = cifar10_model.evaluate(x_test, y_test, verbose=0)
slr_accuracy = score[1]
print('Params:', tU.size + tS.size + tV.size, ', Accuracy:', slr_accuracy)

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

# Load the pre-trained model (ensure you have 'cifar10vgg.json' and 'cifar10vgg.weights.h5' in the directory)
json_file = open('cifar10vgg.json', 'r')
cifar10_model_json = json_file.read()
json_file.close()
cifar10_model = model_from_json(cifar10_model_json)
cifar10_model.load_weights("cifar10vgg.weights.h5")
cifar10_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Get the FC6 Layer Weights and Biases
fc1 = cifar10_model.layers[fc_id].get_weights()
weights = fc1[0]  # The weights matrix
biases = fc1[1]   # The biases vector

# Perform SVD on the weights matrix
U, S, V = np.linalg.svd(weights, full_matrices=False)

# Apply sparse low-rank factorization to the weights
tU, tS, tV = U[:, 0:rank], S[0:rank], V[0:rank, :]

# Reconstruct the compressed weights matrix
compressed_weights = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Set the weights of the layer (preserving the biases)
fc1[0] = compressed_weights
cifar10_model.layers[fc_id].set_weights(fc1)

# Evaluate the model
score = cifar10_model.evaluate(x_train, y_train, verbose=0)
svd_cost = score[0]

# Apply further sparsity to the weights
subrank = int(rank * rr)

# Process sparsity in chunks, reducing the frequency of evaluation
# We will sparsify in chunks of 10 rows and only evaluate every 10th iteration
for ind in range(0, U.shape[0], 10):
    # Apply sparsity to the U matrix in chunks (instead of row by row)
    tempU = np.copy(tU)  # Only copy when we need to update
    tempU[ind:ind+10, subrank:] = 0  # Apply sparsity in batches of 10 rows

    # Reconstruct the sparse weights
    sparse_weights = np.matmul(np.matmul(tempU, np.diag(tS)), tV)
    
    # Update the weights again with the sparsified weights
    fc1[0] = sparse_weights
    cifar10_model.layers[fc_id].set_weights(fc1)
    
    # Evaluate the model only every 10th iteration (to speed up the process)
    if ind % 500 == 0:  # Evaluate after every 100 iterations (adjust based on your needs)
        score = cifar10_model.evaluate(x_train, y_train, verbose=0)
        svd_cost_t = score[0]
        print(f"Evaluation at iteration {ind}: Loss = {svd_cost_t}")

# Final evaluation after sparsification
score = cifar10_model.evaluate(x_test, y_test, verbose=0)
slr_accuracy = score[1]
print('Params:', tU.size + tS.size + tV.size, ', Accuracy:', slr_accuracy)

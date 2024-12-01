from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
import numpy as np

class cifar10vgg:
    def __init__(self, train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg_new.h5')

    def build_model(self):
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=self.x_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        # Add other layers following the same structure...
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes))
        model.add(layers.Activation('softmax'))

        return model

    def normalize(self, X_train, X_test):
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def train(self, model):
        batch_size = 128
        maxepochs = 500
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        half_train_size = x_train.shape[0] // 2
        x_train, y_train = x_train[:half_train_size], y_train[:half_train_size]

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

        datagen.fit(x_train)

        # Model compilation and training
        model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
        model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=maxepochs)

        return model

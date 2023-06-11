import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_val = tf.keras.utils.normalize(x_val, axis = 1)
    return (x_train, y_train), (x_val, y_val)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = load_data()
    if sys.argv[1] == 'train':
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
        model.add(tf.keras.layers.Dense( 128, activation = 'relu'))
        model.add(tf.keras.layers.Dense( 128, activation = 'relu'))
        model.add(tf.keras.layers.Dense(10 , activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy'])
        model.fit(x_train, y_train, epochs = 5)
        model.save('digit.model')
    if sys.argv[1] == 'validate':
        model = tf.keras.models.load_model('digit.model')
        loss, accuracy = model.evaluate(x_val, y_val)
        print(loss, accuracy)
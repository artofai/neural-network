#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The Art of an Artificial Intelligence
http://art-of-ai.com
https://github.com/artofai
"""

__author__ = 'xevaquor'
__license__ = 'MIT'

import numpy as np
import matplotlib.pyplot as plt
import ml_plot as mp
import random
from Data import Data


class Perceptron(object):
    def __init__(self, initial_weights=None):
        self.w = initial_weights
        self.training_error = None


    def activation(self, v):
        return v > 0

    def train_perceptron(self, TrainingData, learning_rate=0.05, iterations=2000, callback_frequency=10,
                         callback=lambda a, b, c: None, compute_error=False):

        if self.w is None:
            self.w = np.random.uniform(size=TrainingData.features)

        self.training_error = np.zeros((iterations))

        for i in range(iterations):
            index = np.random.randint(0, TrainingData.training_samples)
            x = TrainingData.X[index]
            v = x.dot(self.w)
            actual_y = self.activation(v)
            expected_y = TrainingData.y[index]
            delta = expected_y - actual_y
            self.w += learning_rate * delta * x
            if i % callback_frequency == 0:
                callback(x, self.w, i)

            self.training_error[i] = self.get_error(TrainingData)
            if self.training_error[i] == 0:
                return

        if self.training_error[-1] != 0:
            print('Warning: non-zero error. Consider use more iterations or bigger learning rate.')

    def get_error(self, data):
        predictions = self.activation(np.dot(data.X, self.w))
        error = np.sum(predictions != data.y)
        return error

def on_train(x, w, i):
    global D
    plt.clf()
    mp.init_common([-1,10,-1,10])
    mp.plot_data(D)
    mp.plot_boundary(w)
    mp.plot_normal_vector(w)
    plt.scatter(x[1], x[2], marker=(0, 3, 0), color='yellow', s=240, facecolors='none')
    plt.title("Iteration {} w=[{:.2f}, {:.2f}, {:.2f}]".format(i, w[0], w[1], w[2]))
    plt.savefig("{:05d}.png".format(i))

D = Data.Sample()
p = Perceptron()
p.train_perceptron(D, callback=on_train, callback_frequency=100)
plt.clf()
mp.init_common([-1,10,-1,10])
mp.plot_data(D)
mp.plot_boundary(p.w)
plt.show()
mp.plot_error(p.training_error, D.training_samples)
plt.show()
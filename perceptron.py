#!/usr/bin/python
"""
The Art of an Artificial Intelligence
http://art-of-ai.com
https://github.com/artofai
"""

__author__ = 'xevaquor'
__license__ = 'MIT'

import numpy as np


class Perceptron(object):
    def __init__(self, initial_weights=None):
        self.w = initial_weights
        self.training_error = None

    def activation(self, v):
        return v > 0

    def train_perceptron(self, TrainingData, learning_rate=0.05,
                         iterations=2000, callback_frequency=10,
                         callback=lambda a, b, c: None, compute_error=False):

        """
        Trains perceptron by given data
        :param TrainingData: training data
        :param learning_rate: learning rate (eta)
        :param iterations: maximum number of iterations
        :param callback_frequency: how often call callback function
        :param callback: function called after finishing single iteration
        - empty by default
        :param compute_error:
        :return:
        """

        # if no initial weights set, choose them randomly
        if self.w is None:
            self.w = np.random.uniform(size=TrainingData.features)

        self.training_error = np.zeros((iterations))

        for i in range(iterations):
            # get random row
            index = np.random.randint(0, TrainingData.training_samples)
            # extract features from that row
            x = TrainingData.X[index]
            # compute dot product with weights
            v = x.dot(self.w)
            # compute actual perceptron output
            actual_y = self.activation(v)
            # get expected output
            expected_y = TrainingData.y[index]
            # compute delta
            delta = expected_y - actual_y
            # update weights
            self.w += learning_rate * delta * x

            # call callback func when it is time
            if i % callback_frequency == 0:
                callback(x, self.w, i)

            # check if there is no error - we can safely finish
            self.training_error[i] = self.get_error(TrainingData)
            if self.training_error[i] == 0:
                return

        # optionally if there is still error after leaning, warn
        if self.training_error[-1] != 0:
            pass
            # print('Warning: non-zero error. "
            # " Consider use more iterations or bigger learning rate.')

    def get_error(self, data):
        predictions = self.activation(np.dot(data.X, self.w))
        error = np.sum(predictions != data.y)
        return error

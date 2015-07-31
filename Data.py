#!/usr/bin/python
"""
The Art of an Artificial Intelligence
http://art-of-ai.com
https://github.com/artofai
"""

__author__ = 'xevaquor'
__license__ = 'MIT'

import numpy as np

class Data(object):
    def __init__(self, raw_data, y):
        self.X = np.column_stack((np.ones((raw_data.shape[0], 1.)), raw_data))
        self.y = y
        self.training_samples, self.features = self.X.shape
        self.shape = self.X.shape
        
    @staticmethod
    def from_file(filename):
        data = np.loadtxt(filename)
        return data(data[:, 0:-1], data[:,-1])     
        
    @staticmethod
    def sample():
        positive_data = np.random.normal(size=(200, 2)) + 5
        negative_data = np.random.normal(size=(200, 2))
        data = np.concatenate((positive_data, negative_data))
        labels = np.concatenate((
            positive_data.shape[0] * [1],
            negative_data.shape[0] * [0]
        ))
        return Data(data, labels)
        
    @staticmethod
    def from_two_classes(positives, negatives):
        data = np.concatenate((positives, negatives))
        labels = np.concatenate((
            positives.shape[0] * [1],
            negatives.shape[0] * [0]
        ))
        return Data(data, labels)

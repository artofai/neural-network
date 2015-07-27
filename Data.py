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

class Data(object):
    def __init__(self, raw_data, y):
        self.X = np.column_stack((np.ones((raw_data.shape[0], 1.)), raw_data))
        self.y = y
        self.training_samples, self.features = self.X.shape
        self.shape = self.X.shape

    @staticmethod
    def Sample():
        positive_data = np.random.normal(size=(200, 2)) + 5
        negative_data = np.random.normal(size=(200, 2))
        data = np.concatenate((positive_data, negative_data))
        labels = np.concatenate((
            positive_data.shape[0] * [1],
            negative_data.shape[0] * [0]
        ))
        return Data(data, labels)



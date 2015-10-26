#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The Art of an Artificial Intelligence
http://art-of-ai.com
https://github.com/artofai
"""
from IPython.parallel.client.magics import output_args

__author__ = 'xevaquor'
__license__ = 'MIT'

import numpy as np
import util

class LayerBase(object):
    def __init__(self):
        self.size = None
        self.W = np.zeros((0,0))

    def random_init(self):
        raise NotImplementedError()


class SigmoidOutputLayer(LayerBase):
    def __init__(self, neurons_count, previous_layer_count):
        self.size = neurons_count
        self.prev_layer_size = previous_layer_count
        self.W = np.zeros((self.prev_layer_size + 1, self.size))

    def random_init(self):
        self.W = np.random.normal(size=self.W.shape)

    def phi(self, v):
        return 1. / (1 + np.exp(-v))

    def phi_prime(self, v):
        return np.exp(-v)/((1 + np.exp(-v))**2)

class SigmoidInputLayer(LayerBase):
    def __init__(self, input_size):
        self.size = input_size
        self.W = np.zeros((0,0))

    def random_init(self):
        pass

class SigmoidHiddenLayer(LayerBase):
    def __init__(self, layer_size, prev_layer_size):
        self.size = layer_size
        self.prev_layer_size = prev_layer_size
        self.W = np.zeros((self.prev_layer_size + 1, self.size))

    def phi(self, v):
        return 1. / (1 + np.exp(-v))

    def phi_prime(self, v):
        return np.exp(-v)/((1 + np.exp(-v))**2)

    def random_init(self):
        self.W = np.random.normal(size=self.W.shape)



class NN(object):
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = [SigmoidInputLayer(input_size)]
        for size in hidden_sizes:
            self.layers.append(SigmoidHiddenLayer(size, self.layers[-1].size))
        self.layers.append(SigmoidOutputLayer(output_size, self.layers[-1].size))

    def set_wages(self, wages):
        shapes = list([l.W.shape for l in self.layers[1:]])
        packed = list(util.wrap_matrix(wages, shapes))

        assert len(packed) == len(self.layers) - 1

        for i, layer in enumerate(self.layers[1:]):
            layer.W = packed[i]

    def get_wages(self):
        all_wages = [layer.W for layer in self.layers]
        return util.unwrap_matrix(all_wages)

    def random_init(self):
        for layer in self.layers:
            layer.random_init()

    def forward(self, X):
        # hiden layer
        m, n = X.shape
        # examples, features
        assert n == self.layers[0].size

        # 0->1
        bias = np.ones((m, 1))
        X = np.hstack((bias, X))

        v1 = np.dot(X, self.layers[1].W)
        a1 = self.layers[1].phi(v1)

        # 1->2
        bias = np.ones((a1.shape[0], 1))
        a1 = np.hstack((bias, a1))
        v2 = np.dot(a1, self.layers[2].W)
        a2 = self.layers[2].phi(v2)

        self.y_hat = a2
        return a2


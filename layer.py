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
import util

class LayerBase(object):
    def __init__(self):
        self.size = None
        self.W = np.zeros((0,0))
        self.a = None
        self.v = None

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

        self.layers[0].a = X

        for i in range(1, len(self.layers)):
            source_layer = self.layers[i-1]
            dest_layer = self.layers[i]

            bias = np.ones((source_layer.a.shape[0], 1))
            source_layer.a = np.hstack((bias, source_layer.a))

            dest_layer.v = np.dot(source_layer.a, dest_layer.W)
            dest_layer.a = dest_layer.phi(dest_layer.v)

        self.y_hat = self.layers[-1].a
        return self.y_hat

    def cost(self, X, y):
        self.y_hat = self.forward(X)
        J = 0.5*np.sum((y-self.y_hat)**2)
        return J

    def nabla_cost(self, X, y):
        self.forward(X)
        return (self.y_hat - y)

    def cost_prime(self, X, y):

        nabla_cost = self.nabla_cost(X, y)
        delta2 = np.multiply(nabla_cost, self.layers[-1].phi_prime(self.layers[-1].v))
        truncatedW = self.layers[-1].W[1:, :]
        #truncatedW = self.layers[-1].W

        delta1 = np.multiply(np.dot(delta2, truncatedW.T), self.layers[1].phi_prime(self.layers[1].v))
        DJdW2 = np.dot(self.layers[1].a.T, delta2)
        bias = np.ones((X.shape[0], 1))
        biased = np.hstack((bias, X))
        dJdW1 = np.dot(biased.T, delta1)

        return dJdW1, DJdW2

        nabla = self.nabla_cost(X, y)
        delta2 = np.multiply(nabla, self.phi_prime(self.v2))

        truncatedW2 = self.W2[1:,:]

        delta1 = np.multiply(np.dot(delta2, truncatedW2.T), self.phi_prime(self.v1))

        dJdW2 = np.dot(self.a1.T, delta2)
        bias = np.ones((self.m, 1))
        X = np.hstack((bias, X))
        dJdW1 = np.dot(X.T, delta1)

        return dJdW1, dJdW2

if __name__ == '__main__':
    dd = NN(5, [2,3,4], 3)
    dd.random_init()

    X = np.array([[1,2,3,4,5],
                  [10,20,30,40,50],
                  [8,6,4,2,4]],dtype=float)

    Y = np.array([[1,0,1],
                 [1,10,3],
                 [1,-4,4]], dtype=float)

    yyy = dd.forward(X)
    #print(yyy)

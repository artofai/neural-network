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

class NeuralNetwork(object):
    def __init__(self):
        # set network configuration
        self.input_layer_size = 2
        self.hidden_layer_size = 8
        self.output_layer_size = 5
        self.random_init()

    # activation function
    def phi(self, v):
        return 1. / (1 + np.exp(-v))

    def phi_prime(self, v):
        return np.exp(-v)/((1 + np.exp(-1))**2)

    def cost(self, X, y):
        self.y_hat = self.forward(X)
        J = 0.5*sum((y-self.y_hat)**2)
        return J

    def nabla_cost(self, X, y):
        self.forward(X)
        return (self.y_hat - y)

    def cost_prime(self, X, y):
        self.forward(X)

        assert X.shape == (self.m, self.n)
        assert y.shape == (self.m, self.output_layer_size)

        delta2 = np.multiply(-(y - self.y_hat), self.phi_prime(self.v2))

        assert delta2.shape == (self.m, self.output_layer_size)

        dJdW2 = np.dot(self.a1.T, delta2)

        assert dJdW2.shape == (self.hidden_layer_size + 1, self.output_layer_size)

        assert self.v1.shape == (self.m, self.hidden_layer_size)

        truncatedW2 = self.W2[1:,:]
        assert truncatedW2.T.shape == (self.output_layer_size, self.hidden_layer_size)

        delta1 = np.multiply( np.dot( delta2, truncatedW2.T )    ,self.phi_prime(self.v1))

        assert delta1.shape == (self.m, self.hidden_layer_size)
        bias = np.ones((self.m, 1))
        X = np.hstack((bias, X))
        assert X.shape == (self.m, self.n + 1)
        dJdW1 = np.dot(X.T, delta1)
        assert dJdW1.shape == (self.input_layer_size + 1, self.hidden_layer_size)

        return dJdW1, dJdW2

    def alternative_cost_prime(self, X, y):
        self.forward(X)

        nabla = self.nabla_cost(X, y)
        delta2 = np.multiply(nabla, self.phi_prime(self.v2))

        truncatedW2 = self.W2[1:,:]

        delta1 = np.multiply(np.dot(delta2, truncatedW2.T), self.phi_prime(self.v1))

        dJdW2 = np.dot(self.a1.T, delta2)
        bias = np.ones((self.m, 1))
        X = np.hstack((bias, X))
        dJdW1 = np.dot(X.T, delta1)

        return dJdW1, dJdW2

    # initialize weights randomly
    def random_init(self):
        self.W1 = np.random.normal(1,size=(
            self.input_layer_size + 1, self.hidden_layer_size))

        self.W2 = np.random.normal(size=(
           self.hidden_layer_size + 1, self.output_layer_size))

    # compute value for a set of cases
    def forward(self, X):
        # hiden layer
        self.m, self.n = X.shape
        bias = np.ones((self.m, 1))
        X = np.hstack((bias, X))
        self.v1 = np.dot(X, self.W1)
        self.a1 = self.phi(self.v1)

        # output layer
        bias = np.ones((self.a1.shape[0], 1))
        self.a1 = np.hstack((bias, self.a1))
        self.v2 = np.dot(self.a1, self.W2)
        self.a2 = self.phi(self.v2)

        self.y_hat = self.a2
        return self.y_hat


X = np.array([[2,3]])

y = np.array([[0,0,0,0,0]])

nn = NeuralNetwork()
old = nn.cost_prime(X, y)
new = nn.alternative_cost_prime(X, y)

print(old[0] - new[0])
print(old[1] - new[1])


#print(y)


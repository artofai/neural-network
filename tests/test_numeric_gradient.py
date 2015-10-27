from unittest import TestCase

__author__ = 'Xevaquor'

import layer
import numpy as np
import util

def compute_gradients(network, X, y):
    a, b = network.cost_prime(X, y)
    return util.unwrap_matrix([a,b])

def numerical_gradient(network, X, y):
        initial = network.get_wages()
        numgrad = np.zeros(initial.shape)
        peturb = np.zeros(initial.shape)
        epsilion = 1e-4

        for p in range(len(initial)):
            peturb[p] = epsilion
            network.set_wages(initial + peturb)
            loss2 = network.cost(X, y)

            network.set_wages(initial - peturb)
            loss1 = network.cost(X, y)

            numgrad[p] = (loss2 - loss1) / (2. * epsilion)

            peturb[p] = 0

        network.set_wages(initial)

        return numgrad

class TestNN(TestCase):
    def test_get_set_wages_give_same_result(self):
        xsize = 25
        ysize = 10
        nn = layer.NN(xsize, [3,4], ysize)
        X = np.random.uniform(-100,100,size=(1,xsize))
        Y = np.random.uniform(-100,100,size=(1,ysize))
        grad = compute_gradients(nn, X, Y)
        numgrad = numerical_gradient(nn, X, Y)
        score = np.linalg.norm(grad- numgrad)/np.linalg.norm(grad+numgrad)
        print('Gradient check score: ', score)
        self.assertTrue(score < 1e-7)


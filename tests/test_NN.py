from unittest import TestCase

__author__ = 'Xevaquor'

import layer
import numpy as np
import util

class TestNN(TestCase):

    def test_get_set_wages_give_same_result(self):
        nn = layer.NN(5, [3,2,1], 4)
        nn.random_init()
        w = nn.get_wages()
        for l in nn.layers[1:]:
            l.W[0,0] = -17
        nn.set_wages(w)
        w2 = nn.get_wages()

        self.assertTrue(util.list_of_arrays_is_equal(w, w2))


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

def unwrap_matrix(weights):
    out = np.array([])

    for w in weights:
        flat = w.flatten()
        out = np.hstack((out, flat))
        # TODO: ask on stack to o i effetively

    return out

def wrap_matrix(weights, shapes):
    starting_index = 0
    for s in shapes:
        to_take = np.prod(s)
        elements = weights[starting_index:starting_index + to_take]
        starting_index += to_take
       # np.prod
        yield elements.reshape(s)


def test_unwrap():
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    flat = unwrap_matrix([a,b])
    print(flat)
    assert np.array_equal(flat, np.array([1,2,3,4,5,6,7,8]))

def test_wrap():
    flat = np.array(np.linspace(1,26,26))
    nest = wrap_matrix(flat, [(2,3),(4,5)])
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[7,8,9,10,11],
                  [12,13,14,15,16],
                  [17,18,19,20,21],
                  [22,23,24,25,26]])

    r = [x for x in nest]

    assert np.array_equal(r[0], a)
    assert np.array_equal(r[1], b)

test_unwrap()
test_wrap()

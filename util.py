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
    """
    Creates matrices given size, from provided vector of values
    :param weights: Weights to unpack
    :param shapes: Iterable of consecutive matrces sizes
    """
    starting_index = 0
    for s in shapes:
        to_take = np.prod(s)
        elements = weights[starting_index:starting_index + to_take]
        starting_index += to_take
        yield elements.reshape(s)


def list_of_arrays_is_equal(a, b):
    if len(a) != len(b):
        return False

    for i in range(0, len(a)):
        if not np.array_equal(a[i], b[i]):
            return False

    return True

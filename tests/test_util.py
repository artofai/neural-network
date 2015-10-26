from unittest import TestCase

import util
import numpy as np


class TestListOfArraysIsEqual(TestCase):
    def test_are_equal(self):
        a = [np.array([[1, 2, 3], [4, 5, 6]]),
             np.array([[1, 2, 3], [4, 5, 6]])]
        b = [np.array([[1, 2, 3], [4, 5, 6]]),
             np.array([[1, 2, 3], [4, 5, 6]])]

        self.assertTrue(util.list_of_arrays_is_equal(a, b))

    def test_are_not_equal(self):
        a = [np.array([[1, 2, 3], [4, 5, 6]]),
             np.array([[1, 2, 3], [4, 7, 6]])]
        b = [np.array([[1, 2, 3], [4, 5, 6]]),
             np.array([[1, 2, 3], [4, 5, 6]])]

        self.assertFalse(util.list_of_arrays_is_equal(a, b))


class TestWrap(TestCase):
    def test_unwrap(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        flat = util.unwrap_matrix([a, b])
        print(flat)
        assert np.array_equal(flat, np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_wrap(self):
        flat = np.array(np.linspace(1, 26, 26))
        nest = util.wrap_matrix(flat, [(2, 3), (4, 5)])
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9, 10, 11],
                      [12, 13, 14, 15, 16],
                      [17, 18, 19, 20, 21],
                      [22, 23, 24, 25, 26]])

        r = [x for x in nest]

        assert np.array_equal(r[0], a)
        assert np.array_equal(r[1], b)

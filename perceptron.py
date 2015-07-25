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
'''
positive_data = np.array([
    [2.,6],
    [6.,9],
    [5.,8],
    [4.,9],
    [8.,6],
    [4.,5],
    [3.,4],
    [5.,2],
    [8.,3],
    ], dtype=np.double)
    
'''

positive_data = np.random.normal(size=(200, 2)) + 5
negative_data = np.random.normal(size=(200, 2))

positive_data = np.array([
    [3.,7],
    [7.,9],
    [6.,9],
    [5.,9],
    [9.,7],
    [5.,6],
    [4.,5],
    [6.,3],
    [9.,4],
    ], dtype=np.double)
    
negative_data = np.array([
    [0.,5],
    [1.,1],
    [1.,3],
    [4.,2],
    [0.,6],
    [3.,2],
    [0.,1],
    [2.,0],
    [1.,0]
    ], dtype=np.double)

data = np.concatenate((positive_data, negative_data))
labels = np.concatenate((
positive_data.shape[0] * [1],
 negative_data.shape[0] * [0]
                    ))
data = np.column_stack(     (np.ones((data.shape[0], 1.)) , data)    )

m,n = data.shape

errors = np.zeros(1000)
def get_error(w, i):
    global data
    global labels
    global errors
    global m
    err = 0
    for j in range(m):
        actual_y = activation(data[j,:].dot(w))
        expected_y = labels[j]
        #print(actual_y, expected_y)
        if expected_y != actual_y:
            err += 1
    errors[i] = err


def train_perceptron(w, X, y, activation_func, learning_rate = 0.01, 
                     iterations=1000, callback_mod = 10, callback=lambda a, b, c: None):
        
    for i in range(iterations):
        index = np.random.randint(0, m )
        x = X[index]
        v = x.dot(w)
        actual_y = activation_func(v)
        expected_y = y[index]
        delta = expected_y - actual_y
        for feature in range(len(w)):
            w[feature] +=  learning_rate * delta * x[feature]
        if i % callback_mod==0:            
            callback(x, w, i)        
    return w
    

def activation(v):
    return 1 if v > 0 else 0

def outs(ins, ws):
    for i, entry in enumerate(ins):
        v = entry.dot(ws)
        print(i, ': ', entry, 'v=', v, ' y=', activation(v))

plot_range = [-10,10,-10,10]
def on_train(x, w, i):
    global call_nr
    get_error(w, i)    
    mp.init_common(plot_range)
    mp.plot_data(positive_data, negative_data)
    mp.plot_boundary(w)
    mp.plot_normal_vector(w)
    plt.scatter(x[1], x[2], marker=(0,3,0), color='red', s=240, facecolors='none')
    plt.title("Iteration {} w=[{:.2f}, {:.2f}, {:.2f}]".format(i, w[0], w[1], w[2]))
    plt.show()
    

w = np.array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
#w = np.array([1,1,1.])
train_perceptron(w, data, labels, activation, callback=on_train)
plt.plot(np.linspace(1,1000,1000), errors)
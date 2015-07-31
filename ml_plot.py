#!/usr/bin/python
"""
The Art of an Artificial Intelligence
http://art-of-ai.com
https://github.com/artofai
Module in this version does not handle w = [0,0,0] or w = [_, _, 0] properly
"""

__author__ = 'xevaquor'
__license__ = 'MIT'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from data import *
from perceptron import * 
 
def init_common(plot_size=[-5,5,-5,5]):
    plt.grid()
    ax = plt.axes()
    ax.set_aspect('equal')
    plt.axis(plot_size)

def plot_arrow(o, d, head_len=0.3, head_width=0.2, color='black', **kwargs):
    ax = plt.axes()
    angle = np.arctan2(d[1], d[0])
    dx = d[0] - head_len * np.cos(angle)
    dy = d[1] - head_len * np.sin(angle)
    ax.arrow(o[0], o[1], dx, dy,ec=color, fc=color,
             head_width=head_width, head_length = head_len, antialiased=True,
             **kwargs)
             

 
def plot_boundary(w, fill=True, linecolor='black', pos_color='orange',
                  neg_color='purple', fill_alpha=0.2, **kwargs):
    X = np.linspace(-100,100,50)    
    Y = (-w[1] * X - w[0]) / w[2]
    plt.plot(X, Y, color=linecolor, **kwargs)
    
    if not fill:
        return
    
    ax = plt.axes()
    angle = np.arctan2(w[2], w[1])
    if angle < 0: 
        pos_color, neg_color = neg_color, pos_color
    
    ax.fill_between(X, Y, y2=-100, interpolate=True, color=neg_color,
                    alpha=fill_alpha)
    ax.fill_between(X, Y, y2=100, interpolate=True, color=pos_color,
                    alpha=fill_alpha)
 
def plot_normal_vector(w, head_len=0.3, head_width=0.2, color='black',
                       **kwargs):
    d = -w[0] / np.linalg.norm(w[1:])
    angle = np.arctan2(w[2], w[1])    
    xoffset = np.cos(angle) * d
    yoffset = np.sin(angle) * d
    ax = plt.axes()
    if np.abs(head_len) > np.max(np.abs(w[1:])):
        print('[ml_plot] Arrow too $short, omitting')
        # yes, I know it is possible to handle it
        head_len = np.max(np.abs(w[1:]))

    dx = w[1] - head_len * np.cos(angle)
    dy = w[2] - head_len * np.sin(angle)
    ax.arrow(xoffset, yoffset, dx, dy,ec=color, fc=color,
             head_width=head_width, head_length = head_len, antialiased=True,
             **kwargs)

def plot_data(data, marker_size=120):
    positive = data.X[data.y == 1]
    negative = data.X[data.y == 0]

    plt.scatter(negative[:,1], negative[:,2], marker=(0,3,0), facecolors='none',
                s=marker_size, color='purple')
    plt.scatter(positive[:,1], positive[:,2], marker='D', s=marker_size,
                color='orange')
    
def plot_error(error, cases):
    error = error * 100.0 / cases
    ax = plt.axes()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%0.0f%%"))
    #plt.ylim(0,100)
    plt.xlabel("Iteration #")
    plt.ylabel("Total error")
    plt.plot(error, color='red', linewidth=2.0)
    
          
if __name__ == '__main__':
    w = [0,0,2]
    d = Data(np.array([[-2,-1]]), np.array([1]))
    init_common([-3,2,-2,3])
    plot_data(d)
    plot_boundary(w, neg_color='purple', pos_color='orange', fill=False,
                  ls='dashed', linecolor='red')
    
    plt.scatter([-2], [-1], marker=(0, 3, 0), color='blue', s=240, 
                facecolors='red')
    plot_data(d)
    plot_normal_vector(w, ls='dashed', color='red')
    plot_arrow([0,2], [-2,-1], ls='dotted')
    plot_arrow([0,0], [-2,-1], ls='dotted')
    plot_arrow([0,0], [-2,1])
    plot_boundary([0, -2,1], fill=True)

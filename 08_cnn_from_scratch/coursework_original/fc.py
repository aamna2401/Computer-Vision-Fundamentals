# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:31:12 2021

@author: danis
"""


import numpy as np
from layer import Layer


class FullyConnected(Layer):
    def __init__(self, size, activation):
        super().__init__()
        self.size = size
        self.activation = activation
        self.is_softmax = False
        self.cache = {}
        self.w = None
        self.b = None

    def init(self, in_dim):
        # He initialization
        self.w = np.random.randn(self.size, in_dim) * np.sqrt(2 / in_dim)

        self.b = np.zeros((1, self.size))

    def forward(self, a_prev, training):
        z = np.dot(a_prev, self.w.T) + self.b
        a = self.activation.f(z)

        if training:
            # Cache for backward pass
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.size

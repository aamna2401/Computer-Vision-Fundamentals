# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:06:36 2021

@author: danis
"""


class Layer:

    def init(self, in_dim):
        raise NotImplementedError

    def forward(self, a_prev, training):
        raise NotImplementedError

    def backward(self, da):
        raise NotImplementedError

    def update_params(self, dw, db):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

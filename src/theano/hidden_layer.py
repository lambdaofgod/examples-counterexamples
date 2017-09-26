import numpy as np

import theano
import theano.tensor as T
from weight_initialization import initialize_weights


class HiddenLayer:


    def __init__(
            self,
            input_size,
            size,
            activation,
            initialization_type,
            iid):
        self.id = iid
        W = initialize_weights((input_size, size), initialization_type)
        b = initialize_weights((size, ), initialization_type)
        self.W = theano.shared(value=W, name='W_{}'.format(self.id))
        self.b = theano.shared(value=b, name='b_{}'.format(self.id))
        self.activation = activation


    def forward(self, X):
        return self.activation(X.dot(self.W) + self.b)


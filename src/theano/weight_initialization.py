import numpy as np

import theano
import theano.tensor as T


def initialize_weights(shape, initialization_type):
    if initialization_type == 'xavier':
        return np.sqrt(2 / np.sum(shape)) * np.random.randn(*shape)
    else:
        return np.zeros(shape)

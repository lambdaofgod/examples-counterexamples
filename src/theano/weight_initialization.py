import numpy as np


def initialize_weights(shape, initialization_type):
    if initialization_type == 'xavier' and len(shape) > 1:
        return np.sqrt(4 / shape[0]) * np.random.uniform(low=-1, high=1, size=shape)
    else:
        return np.zeros(shape)

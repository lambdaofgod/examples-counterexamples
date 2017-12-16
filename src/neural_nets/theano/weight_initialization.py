import numpy as np


def initialize_weights(shape, initialization_type):
    if initialization_type == 'xavier' and len(shape) > 1:
        return np.sqrt(4 / shape[0]) * np.random.uniform(low=-1, high=1, size=shape)
    elif initialization_type == 'zero' or len(shape) == 1:
        return np.zeros(shape)
    else:
        raise NotImplementedError("{} -type initialization not supportad".format(initialization_type))

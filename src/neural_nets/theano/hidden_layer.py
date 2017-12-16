import theano
from .weight_initialization import initialize_weights


class HiddenLayer:
    def __init__(self,
                 input_size,
                 size,
                 activation,
                 initialization_params,
                 iid):
        self.id = iid
        W, b = HiddenLayer._initial_weights(initialization_params, input_size, size)
        self.W = theano.shared(value=W, name='W_{}'.format(self.id))
        self.b = theano.shared(value=b, name='b_{}'.format(self.id))
        self.activation = activation

    def forward(self, X):
        return self.activation(X.dot(self.W) + self.b)

    @staticmethod
    def _initial_weights(initialization_params, input_size, size):
        if type(initialization_params) == 'dict':
            assert initialization_params.get('W') is not None
            assert initialization_params.get('b') is not None
            W = initialization_params['W']
            b = initialization_params['b']
            assert W.shape[1] == b.shape[0], 'weight shapes not aligned'
        else:
            initialization_type = initialization_params
            W = initialize_weights((input_size, size), initialization_type)
            b = initialize_weights((size,), initialization_type)
        return W, b

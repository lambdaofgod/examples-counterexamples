import numpy as np
import theano
import theano.tensor as T

from .updates import gradient_descent_update, momentum_method_updates, nesterov_method_updates, rmsprop_updates
from .weight_initialization import initialize_weights


class NeuralNet:
    @staticmethod
    def initialized_weights(n_dim, n_classes, initialization_type):
        """
        initialize weights (shared variables)
        """

        # initialize class weights
        out_W = theano.shared(
            value=initialize_weights((n_dim, n_classes), initialization_type),
            name='out_W',
            borrow=True)

        # initialize the biases b as a vector of n_out 0s
        out_B = theano.shared(
            value=initialize_weights((n_classes,), initialization_type),
            name='out_B',
            borrow=True)
        return out_W, out_B

    @staticmethod
    def regularization(W, lmbda, l1_ratio):
        """
        regularization with l1 and l2 weight penalties
        """
        weight_penalty = T.sum(W ** 2)
        l1_penalty = T.sum(abs(W))
        return (lmbda *
                ((1 - l1_ratio) * weight_penalty +
                 l1_ratio * l1_penalty))

    def updates(self, optimization_params):
        """
        choose appropriate updates
        """
        optimization_method = optimization_params['method']
        learning_rate = optimization_params['learning_rate']

        if optimization_method == 'gradient_descent':
            updating_function = gradient_descent_update
        elif optimization_method == 'momentum':
            updating_function = momentum_method_updates
        elif optimization_method == 'nesterov':
            updating_function = nesterov_method_updates
        elif optimization_method == 'rmsprop':
            updating_function = rmsprop_updates
        else:
            raise ValueError("invalid combination of parameters: {}".format(optimization_params))

        if optimization_params.get('decay'):
            decay = optimization_params['decay']
        else:
            decay = None
        return NeuralNet.weight_updates(
            updating_function,
            self.loss,
            self.weights,
            learning_rate,
            decay)

    @staticmethod
    def weight_updates(
        updating_function,
        loss,
        weights_tensors,
        learning_rate,
        decay=None):
        """
        gradient descent updates
        """
        if decay:
            updates = [updating_function(loss, weights, learning_rate, decay)
                       for weights in weights_tensors]
            return sum(updates, [])
        else:
            return ([updating_function(loss, weights, learning_rate)
                     for weights in weights_tensors])

    def iter_training(self, train_model, X, y, n_iter, batch_size):
        """
        iterate weight updates n_iter times and store loss for each step
        """

        def get_batch(batch_size):
            if batch_size:
                indices = np.random.choice(X.shape[0], batch_size, replace=False)
                return X[indices, :], y[indices]
            else:
                return X, y

        self.losses = []
        for __ in range(n_iter):
            X_batch, y_batch = get_batch(batch_size)
            current_loss = train_model(X_batch, y_batch)
            self.losses.append(current_loss)

        self.losses = np.array(self.losses)

        self.is_fitted = True

    def _activation(self, activation_type):
        activations_dict = {
            "tanh": T.tanh,
            "sigmoid": T.nnet.sigmoid,
            "relu": T.nnet.relu
        }
        if activation_type in activations_dict.keys():
            return activations_dict[activation_type]
        else:
            raise NotImplementedError("{} activation unsupported".format(activation_type))
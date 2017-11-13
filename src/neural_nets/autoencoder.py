import numpy as np
import theano
import theano.tensor as T

from neural_nets.weight_initialization import initialize_weights
from .hidden_layer import HiddenLayer
from .neural_net import NeuralNet


class Autoencoder(NeuralNet):
    def __init__(self,
                 n_iter,
                 hidden_sizes,
                 autoencoder_type='standard',
                 activation=T.nnet.relu,
                 initialization_type='xavier',
                 batch_size=1000,
                 lmbda=0.0001,
                 l1_ratio=0,
                 random_state=0,
                 learning_rate=0.001,
                 optimization_params=None,
                 iid='my nnet'):
        """
        Autoencoder

        Autoencoder trained using either full or stochastic gradient descent with
        either vanilla g.d. or momentum learning rule

        Currently only supports training one layer

        Parameters
        ----------
        n_iter : int
            number of iterations

        hidden_sizes : list[int]
            Sizes of hidden layers

        autoencoder_type : string
            Type of autoencoder

        activation : theano function (default relu)
            Activation of neural network

        initialization_type : str (default 'xavier')
            how to initialize weights

        batch_size : int (default=1000)
            number of element in sgd minibatch

        lmbda : float (default 10e-4)
            regularization strength (lambda)

        l1_ratio : float (default 0)
            proportion of l1 to l2 regularization penalty

        random_state : int
            random state used for choosing minibatches

        learning_rate : float (default=0.001)
            learning rate for learning rule

        optimization_params: dict or None (default None)
            what momentum-related rule to use
            for using momentum pass {'method': 'momentum', 'decay': d}
            where d is amount of momentum
        """
        self.id = iid
        self.initialization_type = initialization_type
        self.autoencoder_type = autoencoder_type
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        self.n_iter = n_iter
        self.l1_ratio = l1_ratio
        self.lmbda = lmbda
        self.batch_size = batch_size

        if len(hidden_sizes) != 1:
            raise NotImplementedError("only supports one layer training")

        self.optimization_params = {
            "learning_rate": learning_rate
        }
        if optimization_params:
            self.optimization_params.update(optimization_params)
        else:
            self.optimization_params['method'] = 'gradient_descent'

        np.random.seed(random_state)

        self.is_fitted = False

    def fit(self, X):
        n_dim = X.shape[1]

        # inputs
        thX_in = T.dmatrix('thX_in')
        thX_expected_out = T.dmatrix('thX_expected_out')

        encoder = HiddenLayer(
            n_dim,
            self.hidden_sizes[0],
            self.activation,
            self.initialization_type,
            "{} layer {}".format(self.id, 1))

        decoder_W = encoder.W.T
        decoder_b = theano.shared(
            value=initialize_weights((n_dim, ), self.initialization_type),
            name='b_{}'.format(self.id))

        thH = encoder.forward(thX_in)
        thX_out = self.activation(thH.dot(decoder_W) + decoder_b)
        self.weights = [encoder.W]

        if self.autoencoder_type == 'contractive':
            penalty_term = Autoencoder.contractive_penalty(
                self.activation, thX_in, encoder.W, encoder.b)
            regularization = self.lmbda * penalty_term
        else:
            regularization = T.sum(NeuralNet.regularization(self.weights[0], self.lmbda, self.l1_ratio))

        square_loss = T.mean(
            T.sum(
                (thX_out - thX_expected_out) ** 2,
                axis=1))

        self.loss = square_loss + regularization
        updates = self.updates(self.optimization_params)

        # setup training
        self.train_model = self._training_function(
            thX_in,
            thX_expected_out,
            self.loss,
            updates
        )

        self.iter_training(self.train_model, X, X, self.n_iter, self.batch_size)

        self.retrieve = theano.function(
            inputs=[thX_in],
            outputs=thX_out
        )

        self.project = theano.function(
            inputs=[thX_in],
            outputs=thH
        )

    def transform(self, X):
        return self.project(X)

    def retrieve(self, X):
        return self.retrieve(X)

    def _training_function(self, thX_in, thX_expected_out, loss, updates):
        return theano.function(
            inputs=[thX_in, thX_expected_out],
            outputs=loss,
            updates=updates)

    def contractive_penalty(activation, X, W, b):
        thH_in = X.dot(W) + b
        thH_out = activation(thH_in)
        jacobian = T.grad(thH_out.sum(), thH_in)
        weight_term = W.sum(axis=0)
        return T.sqr(jacobian).dot(T.sqr(weight_term)).sum()

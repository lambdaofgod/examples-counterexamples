import numpy as np
import theano
import theano.tensor as T

from .hidden_layer import HiddenLayer
from .neural_net import NeuralNet


class Autoencoder(NeuralNet):
    def __init__(self,
                 n_iter,
                 hidden_sizes,
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
        if self.batch_size:
            n_examples = self.batch_size
        else:
            n_examples = X.shape[0]

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

        decoder = HiddenLayer(
            self.hidden_sizes[0],
            n_dim,
            self.activation,
            self.initialization_type,
            "{} layer {}".format(self.id, 2))

        thH = encoder.forward(thX_in)
        thX_out = decoder.forward(thH)
        self.encoder_weights = encoder.W
        self.decoder_weights = decoder.W
        self.weights = [self.encoder_weights, self.decoder_weights, encoder.b, decoder.b]

        regularization = T.sum([
                                   NeuralNet.regularization(ws, self.lmbda, self.l1_ratio)
                                   for ws in [self.encoder_weights, self.decoder_weights]])
        square_loss = T.mean(
            T.sum(
                (thX_out - thX_expected_out) ** 2,
                axis=1))

        self.loss = square_loss + regularization
        updates = self.updates(self.optimization_params)

        # setup training
        self.train_model = theano.function(
            inputs=[thX_in, thX_expected_out],
            outputs=self.loss,
            updates=updates)

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

    def retrieve(self):
        return self.retrieve(X)

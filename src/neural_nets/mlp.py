import numpy as np
import theano
import theano.tensor as T
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

from neural_nets.neural_net import NeuralNet
from .hidden_layer import HiddenLayer


class MultilayerPerceptron(NeuralNet):
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
        Multilayer Perceptron

        MLP trained using either full or stochastic gradient descent with
        either vanilla g.d. or momentum learning rule


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

        self.optimization_params = {
            "learning_rate": learning_rate
        }
        if optimization_params:
            self.optimization_params.update(optimization_params)
        else:
            self.optimization_params['method'] = 'gradient_descent'

        np.random.seed(random_state)

        self.is_fitted = False

    def fit(self, X, y):
        if self.batch_size:
            n_examples = self.batch_size
        else:
            n_examples = X.shape[0]

        n_dim = X.shape[1]
        n_classes = len(np.unique(y))

        # inputs
        self.thX = T.dmatrix('thX')
        self.out_y = T.vector('out_y', dtype='int64')

        layers, out, output_size = MultilayerPerceptron.initialized_hidden_layers(
            n_dim,
            self.thX,
            self.hidden_sizes,
            self.activation,
            self.initialization_type,
            self.id
        )

        self.out_W, self.out_B = NeuralNet.initialized_weights(output_size, n_classes, self.initialization_type)

        self.weights = ([layer.W for layer in layers] +
                        [layer.b for layer in layers] +
                        [self.out_W, self.out_B])

        # calculate probability and loss 
        Z = T.dot(out, self.out_W) + self.out_B
        self.p_y_by_x = T.nnet.softmax(Z)
        # negative log likelihood
        ll = (T.log(self.p_y_by_x)
              [T.arange(n_examples), self.out_y])
        nll = - T.mean(ll)

        regularization = T.sum([
                                   NeuralNet.regularization(ws, self.lmbda, self.l1_ratio)
                                   for ws in self.weights])

        self.loss = nll + regularization

        updates = self.updates(self.optimization_params)

        # setup training
        self.train_model = theano.function(
            inputs=[self.thX, self.out_y],
            outputs=self.loss,
            updates=updates)

        self.iter_training(self.train_model, X, y, self.n_iter, self.batch_size)

    def predict(self, X):
        if self.is_fitted:
            return self.__prediction_function()(X)
        else:
            raise NotFittedError

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __prediction_function(self):
        """
        actual function used for predicting y given X
        """
        y_pred = T.argmax(self.p_y_by_x, axis=1)
        return theano.function(
            inputs=[self.thX],
            outputs=y_pred)

    @staticmethod
    def initialized_hidden_layers(n_dim,
                                  input,
                                  hidden_sizes,
                                  activation,
                                  initialization_types,
                                  iid):
        # weights
        tmp_out = input
        layers = []
        input_size = n_dim
        for (n, (size, init_type)) in enumerate(zip(hidden_sizes, initialization_types)):
            layer = HiddenLayer(
                input_size,
                size,
                activation,
                initialization_type,
                "{} layer {}".format(iid, n + 1))
            layers.append(layer)
            tmp_out = layer.forward(tmp_out)
            input_size = size
        return layers, tmp_out, input_size
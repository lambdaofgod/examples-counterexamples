import numpy as np

import theano
import theano.tensor as T

from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score 

from .updates import *
from .hidden_layer import HiddenLayer
from .weight_initialization import initialize_weights


class MultilayerPerceptron:
        
    
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
        
        # weights
        tmp_out = self.thX
        input_size = n_dim
        layers = []
        for (n, size) in enumerate(self.hidden_sizes):
            layer = HiddenLayer(
                    input_size,
                    size,
                    self.activation,
                    self.initialization_type,
                    "{} layer {}".format(self.id, n + 1))
            layers.append(layer)
            tmp_out = layer.forward(tmp_out)
            input_size = size
       
        out = tmp_out

        self.out_W, self.out_B = self.__initialized_weights(input_size, n_classes, self.initialization_type)
        
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
            self.__regularization(ws, self.lmbda, self.l1_ratio)
            for ws in self.weights])
        
        self.loss = nll + regularization
        
        updates = self.__updates(self.optimization_params)
     
        # setup training
        self.train_model = theano.function(
            inputs=[self.thX, self.out_y],
            outputs=self.loss,
            updates=updates)
        
        self.__iter_training(self.train_model, X, y, self.n_iter, self.batch_size)
 

    def predict(self, X):
        if self.is_fitted:
            return self.__prediction_function()(X)
        else:
            raise NotFittedError

            
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
            
             
    def __initialized_weights(self, n_dim, n_classes, initialization_type):
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
             
        
    def __prediction_function(self):
        """
        actual function used for predicting y given X
        """
        y_pred = T.argmax(self.p_y_by_x, axis=1)
        return theano.function(
            inputs=[self.thX],
            outputs=y_pred)
    
    
    def __regularization(self, W, lmbda, l1_ratio):
        """
        regularization with l1 and l2 weight penalties
        """
        weight_penalty = T.sum(W ** 2)
        l1_penalty = T.sum(abs(W))
        return    (lmbda * 
                            ((1 - l1_ratio) * weight_penalty +
                             l1_ratio * l1_penalty))
    
    
    def __updates(self, optimization_params):
        """
        choose appropriate updates
        """
        optimization_method = optimization_params['method']
        learning_rate = optimization_params['learning_rate']
       
        if optimization_method == 'gradient_descent':
            updating_function = gradient_descent_update 
        elif optimization_method == 'momentum':
            updating_function = momentum_method_updates
        elif optimization_method  == 'nesterov':
            updating_function = nesterov_method_updates
        elif optimization_method  == 'rmsprop':
            updating_function = rmsprop_updates
        else:
            raise ValueError("invalid combination of parameters: {}".format(optimization_params))
        
        if optimization_params.get('decay'):
            decay = optimization_params['decay']
        else:
            decay = None
        return MultilayerPerceptron.__weight_updates(
                    updating_function,
                    self.loss,
                    self.weights,
                    learning_rate,
                    decay)

    
    def __weight_updates(
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

        
    def __iter_training(self, train_model, X, y, n_iter, batch_size):
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


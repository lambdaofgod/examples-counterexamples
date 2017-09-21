import numpy as np

import theano
import theano.tensor as T


def gradient_descent_update(loss, weights, learning_rate):
  """
  Updates for gradient descent learning method
  W_{t+1} = W_t - learning_rate * grad(W_t)

  Parameters
  ----------

  loss : theano tensor, scalar function

  weights : theano tensor, model parameters

  learning_rate : scalar
  """
  weights_gradient = T.grad(cost=loss, wrt=weights)
  return (weights, weights - learning_rate * weights_gradient)


def momentum_method_updates(loss, weights, learning_rate, momentum):
  """
  Updates for momentum learning method
  V_{t+1} = momentum * V_t - learning_rate * grad(W_t)
  W_{t+1} = W_t + V_{t+1}
  
  
  Parameters
  ----------

  loss : theano tensor, scalar function

  weights : theano tensor, model parameters

  learning_rate : scalar

  momentum : scalar
  """
  weights_gradient = T.grad(cost=loss, wrt=weights)
  velocity = theano.shared(
    value=np.zeros(
      weights.get_value().shape,
      dtype=theano.config.floatX),
    name='velocity_{}'.format(weights.name))
    
  velocity_update = (velocity, momentum * velocity - learning_rate * weights_gradient)
  weights_update = (weights, weights + velocity)
    
  return [velocity_update, weights_update]

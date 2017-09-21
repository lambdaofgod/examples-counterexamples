def gradient_descent_update(loss, weights, learning_rate):
  weights_gradient = T.grad(cost=loss, wrt=weights)
  return (weights, weights - learning_rate * weights_gradient)

def momentum_method_updates(loss, weights, learning_rate, momentum):
  weights_gradient = T.grad(cost=loss, wrt=weights)
  velocity = theano.shared(
    value=np.zeros(
      weights.get_value().shape,
      dtype=theano.config.floatX),
    name='velocity_{}'.format(weights.name))
    
  velocity_update = (velocity, momentum * velocity - learning_rate * weights_gradient)
  weights_update = (weights, weights + velocity)
    
  return [velocity_update, weights_update]

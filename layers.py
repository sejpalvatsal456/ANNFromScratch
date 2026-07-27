from backend import GPU, xp
from optimizers import Optimizer

class Layer:
  def __init__(self):    
    self.id = None 
    self.input_shape: tuple[int, int] = None
    self.n_nodes: int = None
    self.n_prev_nodes: int = None
    self.trainable = True # false for layer like "Dropout", "Pooling", "Flatten", etc
    self.params = {} # like for dense layer - params = {"W": ..., "b": ...}
    self.grad = {} # like for dense layer - grad = { "W": ..., "b": ... }
  
  def _init_params(self):
    return NotImplementedError
  
  def build(self, input_nodes):
    return NotImplementedError
  
  def forwrad(self, X, training=True):
    return NotImplementedError
  
  def backward(self, dA_prev):
    return NotImplementedError
  
  
class Dense(Layer):
  def __init__(self, n_nodes, act_func, act_deriv=None, input_size: tuple[int, int]=None):    
    super().__init__()
    self.input_size: tuple[int, int] = input_size
    self.act_func = act_func
    self.act_deriv = act_deriv
    self.n_nodes = n_nodes
    self.input = None
    self.Z = None
    self.A = None
    
  
  def _init_params(self):
    self.params["W"] = xp.random.randn(self.n_prev_nodes, self.n_nodes) * xp.sqrt(2/self.n_prev_nodes)
    self.params["b"] = xp.zeros((1, self.n_nodes)) 
  
  def build(self, input_nodes):
    self.n_prev_nodes = input_nodes
    self._init_params()
        
  
  def forward(self, X, training=True):
    self.input = X
    self.Z = X @ self.params['W'] + self.params['b']
    self.A = self.act_func(self.Z)
    return self.A
  
  def backward(self, dA_prev):
    # dZ(l) = dA(l-1)*act_seriv(Z(l))
    dZ = dA_prev if self.act_deriv is None else dA_prev * self.act_deriv(self.Z)
    m, _ = self.input.shape
    # dW(l) = (1/m)* X(l).T @ dZ(l)
    # db = (1/m) sum(dZ(l))
    self.grad['W'] = (1/m)* (self.input.T @ dZ)
    self.grad['b'] = (1/m) * xp.sum(dZ, keepdims=True, axis=0)
    
    # dA(l) = dZ(l) @ W.T
    dA = dZ @ self.params['W'].T
    return dA
    


class Dropout(Layer):
  def __init__(self, rate=0.5):
    super().__init__()
    self.rate = rate
    self.trainable = False
    self.mask = None
    
  # def _init_params(self): <- you need this?? bs
    
  def build(self, input_nodes):
    self.n_prev_nodes = input_nodes
    self.n_nodes = input_nodes
        
  def forward(self, X, training=True):
    if not training:
      return X
    self.mask = (xp.random.rand(*X.shape) > self.rate) / (1-self.rate)
    return X * self.mask
  
  def backward(self, dA_prev):
    return self.mask * dA_prev
  
class Flatten(Layer):
  def __init__(self):
    super().__init__()
    self.trainable = False
    self.input_shape = None
    
  def build(self, input_nodes):
    self.n_prev_nodes = input_nodes
    self.n_nodes = input_nodes

  def forward(self, X, training=True):
    self.input_shape = X.shape
    return X.reshape(X.shape[0], -1)

  def backward(self, dA):
    return dA.reshape(self.input_shape)
from backend import GPU, xp
from optimizers import Optimizer

class Layer:
  def __init__(self):    
    self.id = None 
    self.trainable = True # false for layer like "Dropout", "Pooling", "Flatten", etc
    self.params = {} # like for dense layer - params = {"W": ..., "b": ...}
    self.grad = {} # like for dense layer - grad = { "W": ..., "b": ... }
  
  def forwrad(self, X, training=True):
    return NotImplementedError
  
  def backward(self):
    return NotImplementedError
  
  
class Dense(Layer):
  def __init__(self, n_nodes, n_prev_nodes, act_func, act_deriv=None):
    # super().__init__(n_nodes, n_prev_nodes, act_func, act_deriv)
    
    # self.n_nodes = n_nodes
    # self.n_prev_nodes = n_prev_nodes
    
    # self.W = xp.random.randn(n_prev_nodes, n_nodes) * xp.sqrt(2 / n_prev_nodes)
    # self.b = xp.zeros((1, n_nodes))
    # self.input = None
    # self.A = None
    # self.Z = None
    # self.id = None
    
    super().__init__()
    self.act_func = act_func
    self.act_deriv = act_deriv
    self.params["W"] = xp.random.randn(n_prev_nodes, n_nodes) * xp.sqrt(2/n_prev_nodes)
    self.params["b"] = xp.zeros((1, n_nodes))
    self.input = None
    self.Z = None
    self.A = None
  
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
    
    
  # depreciated
  # def update_params(self, dw, db, alpha, optimizer:Optimizer=None):
  #   if not optimizer:
  #     self.W = self.W - alpha*dw
  #     self.b = self.b - alpha*db
  #     return
  #   # else:
  #   #   print("Invalid optimiezr: " + optimizer.name)
  #   #   exit()
  #   self.W = optimizer.update(key=f"W{self.id}", lr=alpha, param=self.W, grad=dw)
  #   self.b = optimizer.update(key=f"B{self.id}", lr=alpha, param=self.b, grad=db)


class Dropout(Layer):
  def __init__(self, n_nodes, n_prev_nodes):
    super().__init__(n_nodes, n_prev_nodes, act_func=None, act_deriv=None)
    
  def calculate(self, X):
    return super().calculate(X)
    
    
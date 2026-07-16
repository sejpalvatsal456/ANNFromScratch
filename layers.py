from backend import GPU, xp
from optimizers import Optimizer

class Layer:
  def __init__(self, n_nodes, n_prev_nodes, act_func, act_deriv=None):
    self.n_nodes = n_nodes
    self.n_prev_nodes = n_prev_nodes
    self.act_func = act_func
    self.act_deriv = act_deriv
  
  def calculate(self, X):
    return NotImplementedError
  
  
  
class Dense(Layer):
  def __init__(self, n_nodes, n_prev_nodes, act_func, act_deriv=None):
    super().__init__(n_nodes, n_prev_nodes, act_func, act_deriv)

    self.W = xp.random.randn(n_prev_nodes, n_nodes) * xp.sqrt(2 / n_prev_nodes)
    self.b = xp.zeros((1, n_nodes))
    self.input = None
    self.A = None
    self.Z = None
    self.id = None

  def calculate(self, X):
    self.input = X
    self.Z = X @ self.W + self.b
    self.A = self.act_func(self.Z)
    return self.A
  
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

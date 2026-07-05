from backend import xp

# Optimizers

class Optimizer:
  def __init__(self, name: str):
    self.name = name
  
  def update(self):
    raise NotImplementedError

class Momentum(Optimizer):
  def __init__(self, beta=0.9):
    super().__init__("Momentum")
    self.beta = beta
    self.velocity = {}
    
  def update(self, key, param, grad, lr):
    # theta_t = beta*theta_(t-1) - lr*grad
    
    if key not in self.velocity:
      self.velocity[key] = xp.zeros_like(param)
    
    self.velocity[key] = self.beta*self.velocity[key] + (1-self.beta)*grad
    return param - lr*self.velocity[key]
    
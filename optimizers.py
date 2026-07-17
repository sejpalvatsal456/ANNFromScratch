from backend import xp

# Optimizers

class Optimizer:
  def __init__(self, name: str,):
    self.name = name
  
  def step(self, model, X, y, lr, epoch):
    raise NotImplementedError
    

class SGD(Optimizer):
  def __init__(self):
    super().__init__("SGD")
  
  def step(self, lr, epoch, model, X=None, y=None ):
    
    for layer in model.layers:
      if not layer.trainable:
        continue
      
      for key in layer.params:
        
        # update params - w(t+1) = w(t) - lr*grad
        layer.params[key] -= lr*layer.grad[key]
      

class Momentum:

  def __init__(self, beta=0.9):
    self.beta = beta
    self.velocity = {}
  
  def step(self, model, lr, epoch, X=None, y=None):
    
    for layer in model.layers:
      if not layer.trainable:
        continue
      
      if layer.id not in self.velocity:
        # v(0) = 0
        self.velocity[layer.id] = {
          "W": xp.zeros_like(layer.params['W']),
          "b": xp.zeros_like(layer.params['b'])
        }
      
      for key in layer.params:
        # get the velocity terms
        # v(t+1) = beta*v(t) + (1-beta)*grad
        self.velocity[layer.id][key] = self.beta*self.velocity[layer.id][key] + (1-self.beta)*layer.grad[key]
        # update params
        # w(t+1) = w(t) - lr*v(t+1)
        layer.params[key] -= lr*self.velocity[layer.id][key]


class NAG(Optimizer):
  def __init__(self, beta=0.9):
    super().__init__("NAG")
    self.beta = beta
    self.velocity = {}
    

  def step(self, model, X, y, lr, epoch):
    
    trainable_layers = [l for l in model.layers if l.trainable]
    saved = []
    for layer in trainable_layers:
      saved.append({ k:v.copy() for k, v in layer.params.items() })
      if layer.id not in self.velocity:
        # v(0) = 0
        self.velocity[layer.id] = {
          k: xp.zeros_like(v) for k, v in layer.params.items()
        }
      
      # w_la = w(t) - beta*v(t-1)
      for key in layer.params:
        layer.params[key] -= self.beta*self.velocity[layer.id][key] 
      
    pred = model.forward_prop(X, training=True)
    model.backward_prop(y_train=y, y_pred=pred)
    
    # get the original params
    for layer, snapshot in zip(trainable_layers, saved):
      for key, value in snapshot.items():
        layer.params[key] = value
      
    # calculate the updated velocity and final params
    for layer in trainable_layers:
      for key in layer.params:
        grad = layer.grad[key]
        # v(t) = beta*v(t-1) + lr*grad(w_la)
        self.velocity[layer.id][key] = self.beta*self.velocity[layer.id][key] + lr*grad
        # w(t+1) = w(t) - v(t)
        layer.params[key] -= self.velocity[layer.id][key]
      
      

class RMSProp(Optimizer):
  def __init__(self, beta=0.95, epsilon=1e-8):
    super().__init__("RMSProp")
    self.beta = beta
    self.epsilon = epsilon
    self.velocity = {}
    
  def step(self, model, X, y, lr, epoch):
    
    for layer in model.layers:
      if not layer.trainable:
        continue
      
      # init velocity to v(0) = 0
      if layer.id not in self.velocity:
        self.velocity[layer.id] = { k: xp.zeros_like(v) for k, v in layer.params.items() }
      
      for key in layer.params:
        # set v(t) = beta*v(t-1) + (1-beta)*grad^2
        self.velocity[layer.id][key] = self.beta*self.velocity[layer.id][key] + (1-self.beta)*(layer.grad[key]**2)
        # update params w(t+1) = w(t) - lr/(sqrt(v(t)) + epsilon) * grad
        layer.params[key] -= lr*layer.grad[key]/(xp.sqrt(self.velocity[layer.id][key]) + self.epsilon)
    
      
class Adam(Optimizer):
  def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
    super().__init__("Adam")
    self.beta1 = beta1
    self.beta2 = beta2 
    self.epsilon = epsilon
    self.momentum = {}
    self.velocity = {}
    self.t = 1
    
  def step(self, model, X, y, lr, epoch):
    
    for layer in model.layers:
      
      if not layer.trainable:
        continue
      
      # v(0) = 0 and m(0) = 0
      if layer.id not in self.velocity:
        self.velocity[layer.id] = { k: xp.zeros_like(v) for k, v in layer.params.items() }
      if layer.id not in self.momentum:
        self.momentum[layer.id] = { k: xp.zeros_like(v) for k, v in layer.params.items() }
      
      for key in layer.params:
        # update momentum - m(t) = beta1*m(t-1) + (1-beta1)*grad  
        self.momentum[layer.id][key] = self.beta1*self.momentum[layer.id][key] + (1-self.beta1)*layer.grad[key]
        
        # update velocity - v(t) = beta2*v(t-1) + (1-beta2)*(grad^2)
        self.velocity[layer.id][key] = self.beta2*self.velocity[layer.id][key] + (1-self.beta2)*(layer.grad[key]**2)
        
        # get the bias coorection
        # m_hat(t) = m(t)/(1-beta1^t)
        m_hat = self.momentum[layer.id][key]/(1-self.beta1**self.t)
        
        # v_hat(t) = v(t)/(1-beta2^t)
        v_hat = self.velocity[layer.id][key]/(1-self.beta2**self.t)
        
        # update the params - w(t+1) = w(t) - lr*m_hat/(sqrt(v_hat) + epsilon)
        layer.params[key] -= lr*m_hat/(xp.sqrt(v_hat) + self.epsilon)
        
    self.t += 1
      
        
        
      
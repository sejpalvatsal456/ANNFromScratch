from backend import xp

# Optimizers

class Optimizer:
  def __init__(self, name: str,):
    self.name = name
  
  def step(self, model, grads, X, y, lr, epoch):
    raise NotImplementedError
    

class SGD(Optimizer):
  def __init__(self):
    super().__init__("SGD")
  
  def step(self, model, grads, X, y, lr):
    for layer, dW, db in grads:
      layer.W -= lr*dW
      layer.b -= lr*db 

class Momentum:

  def __init__(self, beta=0.9):
    self.beta = beta
    self.velocity = {}
  
  def step(self, model, grads, X, y, lr, epoch):
    for layer, dW, db in grads:
      
      if layer.id not in self.velocity:
        # v(0) = 0
        self.velocity[layer.id] = {
          "W": xp.zeros_like(layer.W),
          "b": xp.zeros_like(layer.b)
        }
      
      # get the velocity terms
      # v(t+1) = beta*v(t) + (1-beta)*grad
      self.velocity[layer.id]["W"] = self.beta*self.velocity[layer.id]["W"] + (1-self.beta)*dW
      self.velocity[layer.id]["b"] = self.beta*self.velocity[layer.id]["b"] + (1-self.beta)*db
      
      # update params
      # w(t+1) = w(t) - lr*v(t+1)
      layer.W -= lr*self.velocity[layer.id]["W"]
      layer.b -= lr*self.velocity[layer.id]["b"]
      

class NAG(Optimizer):
  def __init__(self, beta=0.9):
    super().__init__("NAG")
    self.beta = beta
    self.velocity = {}
    

  def step(self, model, grads, X, y, lr, epoch):
    
    # calculate look ahead params
    saved = []
    for layer, dW, db in grads:
      saved.append((layer.W.copy(), layer.b.copy()))
      if layer.id not in self.velocity:
        self.velocity[layer.id] = {
          "W": xp.zeros_like(layer.W),
          "b": xp.zeros_like(layer.b)
        }
      
      # w_la = w(t) - beta*v(t-1)
      layer.W -= self.beta*self.velocity[layer.id]["W"]
      layer.b -= self.beta*self.velocity[layer.id]["b"]
      
    pred = model.forward_prop(X)
    new_grads = model._backward_prop(y_train=y, y_pred=pred, m=len(X))
    
    # get the original params
    for layer, (W, b) in zip(model.layers, saved):
      layer.W = W
      layer.b = b
    
    # calculate velocity
    for layer, dW, db in new_grads:
      
      # v(t) = beta*v(t-1) + lr*grad(w_la)
      self.velocity[layer.id]["W"] = self.beta*self.velocity[layer.id]["W"] + lr*dW
      self.velocity[layer.id]["b"] = self.beta*self.velocity[layer.id]["b"] + lr*db

      # w(t+1) = w(t) - lr*v(t)
      layer.W -= self.velocity[layer.id]["W"]
      layer.b -= self.velocity[layer.id]["b"]


class RMSProp(Optimizer):
  def __init__(self, beta=0.95, epsilon=1e-8):
    super().__init__("RMSProp")
    self.beta = beta
    self.epsilon = epsilon
    self.velocity = {}
    
  def step(self, model, grads, X, y, lr, epoch):
    
    for layer, dW, db in grads:
      
      # init velocity to v(0) = 0
      if layer.id not in self.velocity:
        self.velocity[layer.id] = {
          "W" : xp.zeros_like(layer.W),
          "b" : xp.zeros_like(layer.b)
        }
        
      # set v(t) = beta*v(t-1) + (1-beta)*grad^2
      self.velocity[layer.id]["W"] = self.beta*self.velocity[layer.id]["W"] + (1-self.beta)*(dW**2)
      self.velocity[layer.id]["b"] = self.beta*self.velocity[layer.id]["b"] + (1-self.beta)*(db**2)
      
      # update params w(t+1) = w(t) - lr/(sqrt(v(t)) + epsilon) * grad
      layer.W -= lr*dW/(xp.sqrt(self.velocity[layer.id]["W"]) + self.epsilon)
      layer.b -= lr*db/(xp.sqrt(self.velocity[layer.id]["b"]) + self.epsilon)
      
class Adam(Optimizer):
  def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
    super().__init__("Adam")
    self.beta1 = beta1
    self.beta2 = beta2 
    self.epsilon = epsilon
    self.momentum = {}
    self.velocity = {}
    self.t = 1
    
  def step(self, model, grads, X, y, lr, epoch):
    
    for layer, dW, db in grads:
      
      # init v(0) = 0 and m(0) = 0
      if layer.id not in self.momentum:
        self.momentum[layer.id] = {
          "W": xp.zeros_like(layer.W),
          "b" : xp.zeros_like(layer.b)
        }
      
      if layer.id not in self.velocity:
        
        self.velocity[layer.id] = {
          "W": xp.zeros_like(layer.W),
          "b" : xp.zeros_like(layer.b)
        }
        
      
      # update momentum - m(t) = beta1*m(t-1) + (1-beta1)*grad  
      self.momentum[layer.id]["W"] = self.beta1*self.momentum[layer.id]["W"] + (1-self.beta1)*dW
      self.momentum[layer.id]["b"] = self.beta1*self.momentum[layer.id]["b"] + (1-self.beta1)*db
      
      # update velocity - v(t) = beta2*v(t-1) + (1-beta2)*(grad^2)
      self.velocity[layer.id]["W"] = self.beta2*self.velocity[layer.id]["W"] + (1-self.beta2)*dW**2
      self.velocity[layer.id]["b"] = self.beta2*self.velocity[layer.id]["b"] + (1-self.beta2)*db**2
      
      # get the bias coorection
      # m_hat(t) = m(t)/(1-beta1^t)
      m_hat_W = self.momentum[layer.id]["W"] / (1 - self.beta1**self.t)
      m_hat_b = self.momentum[layer.id]["b"] / (1 - self.beta1**self.t)
      
      # v_hat(t) = v(t)/(1-beta2^t)
      v_hat_W = self.velocity[layer.id]["W"] / (1 - self.beta2**self.t)
      v_hat_b = self.velocity[layer.id]["b"] / (1 - self.beta2**self.t)
      
      # update the params
      layer.W -= lr*m_hat_W/(xp.sqrt(v_hat_W) + self.epsilon)
      layer.b -= lr*m_hat_b/(xp.sqrt(v_hat_b) + self.epsilon)
      self.t += 1
      
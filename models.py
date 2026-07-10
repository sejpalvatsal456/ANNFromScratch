import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from backend import GPU, xp
from optimizers import Optimizer

def one_hot_encoding(y, num_classes=10):
  one_hot = xp.zeros((len(y), num_classes))
  one_hot[xp.arange(len(y)), y] = 1
  return one_hot

def ReLu(X):
  return xp.maximum(0, X)

def ReLu_derive(X):
  return (X > 0).astype(xp.float32)

def softmax(X):
  exp = xp.exp(X - xp.max(X, axis=1, keepdims=True))
  return exp / xp.sum(exp, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    y_pred : (m, n_classes)
    y_true : integer labels (m,)
    """
    m = y_pred.shape[0]

    eps = 1e-12
    y_pred = xp.clip(y_pred, eps, 1 - eps)

    return -xp.mean(xp.log(y_pred[xp.arange(m), y_true]))


class Model:
  def __init__(self, iterations, alpha, optimizer:Optimizer, batch_size=64, decay=0.001):
    self.iterations = iterations
    self.alpha = alpha
    self.alpha0 = alpha
    self.decay = decay
    self.layers = []
    self.batch_size = batch_size
    self.optimizer = optimizer
  
  def add_layer(self, layer):
    layer.id = len(self.layers)
    self.layers.append(layer)
    
  def forward_prop(self, X):
    curr_A = X
    for layer in self.layers:
      # print(layer.W.shape)
      curr_A = layer.calculate(curr_A)
      # print("Curr_A:")
      # print(curr_A)
    return curr_A
    
  def predict(self, X):
    y_pred = self.forward_prop(X)
    return xp.argmax(y_pred, axis=1)
  
  def accuracy(self, X, y):
    predictions = self.predict(X)
    acc = xp.mean(predictions == y)

    return float(acc.get()) if GPU else float(acc)
  
  def _backward_prop(self, y_train, y_pred, m):
    one_hot_y = one_hot_encoding(y_train)
    dZ = y_pred - one_hot_y

    grads = []

    for i in reversed(range(len(self.layers))):
      layer = self.layers[i]

      dW = (layer.input.T @ dZ) / m
      db = xp.sum(dZ, axis=0, keepdims=True) / m

      grads.append((layer, dW, db))
      
      if i != 0:
        prev_layer = self.layers[i - 1]
        dA_prev = dZ @ layer.W.T
        dZ = dA_prev * prev_layer.act_deriv(prev_layer.Z)

    
    grads.reverse()
    return grads
  
  def fit(self, X_train, y_train):
    loss_history = []
    acc_history = []
    iteration_history = []

    for i in range(self.iterations):

      self.alpha = self.alpha0 / (1 + self.decay * i)

      # Shuffle dataset
      indices = xp.random.permutation(len(X_train))
      X_train = X_train[indices]
      y_train = y_train[indices]

      for start in range(0, len(X_train), self.batch_size):

        end = start + self.batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Forward pass
        y_pred = self.forward_prop(X_batch)

        # Loss
        loss = cross_entropy_loss(y_pred, y_batch)
        loss = float(loss.get()) if GPU else float(loss)
        loss_history.append(loss)

        # Accuracy
        pred = xp.argmax(y_pred, axis=1)
        acc = xp.mean(pred == y_batch)
        acc = float(acc.get()) if GPU else float(acc)
        acc_history.append(acc)

        iteration_history.append(i)

        # Compute gradients
        grads = self._backward_prop(y_batch, y_pred, len(X_batch))
        self.optimizer.step(model=self, grads=grads, X=X_batch, y=y_batch, lr=self.alpha, epoch=i+1)
        
      if i % 10 == 0:
        pred = self.predict(X_train)
        acc = xp.mean(pred == y_train)
        acc = float(acc.get()) if GPU else float(acc)
        print(f"Iteration {i} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    if GPU:
        loss_history = xp.asnumpy(xp.array(loss_history))
        acc_history = xp.asnumpy(xp.array(acc_history))
    else:
        loss_history = xp.array(loss_history)
        acc_history = xp.array(acc_history)

    plt.figure(figsize=(8, 5))
    plt.plot(iteration_history, loss_history, marker='o')
    plt.plot(iteration_history, acc_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()
    
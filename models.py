import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from backend import GPU, xp
from optimizers import Optimizer
from layers import Layer
import math

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
  def __init__(self, layers: list[Layer], epochs, alpha, optimizer:Optimizer, input_shape=None, batch_size=64, decay=0.001):
    self.epochs = epochs
    self.alpha = alpha
    self.alpha0 = alpha
    self.decay = decay
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.input_shape = input_shape
    
    self.layers: list[Layer] = []
    if input_shape is None:
      raise ValueError("Model requires an input_shape.")
    prev_nodes = math.prod(input_shape)
    for layer in layers:
      layer.id = len(self.layers)
      layer.build(prev_nodes)
      prev_nodes = layer.n_nodes
      self.layers.append(layer)
  
  # depreciated
  # def add_layer(self, layer):
  #   layer.id = len(self.layers)
  #   self.layers.append(layer)
  
    
  def forward_prop(self, X, training=True):
    curr_A = X
    for layer in self.layers:
      curr_A = layer.forward(curr_A, training=training)
    return curr_A
    
  def predict(self, X):
    y_pred = self.forward_prop(X)
    return xp.argmax(y_pred, axis=1)
  
  def accuracy(self, X, y):
    predictions = self.predict(X)
    acc = xp.mean(predictions == y)

    return float(acc.get()) if GPU else float(acc)
  
  def backward_prop(self, y_train, y_pred):
    one_hot_y = one_hot_encoding(y_train)
    # dZ = y_pred - one_hot_y
    dA = y_pred - one_hot_y

    for layer in reversed(self.layers):
      # dA(l) = backward(dA(l-1))
      dA = layer.backward(dA_prev=dA)
  
  def fit(self, X_train, y_train):
    loss_history = []
    acc_history = []
    iteration_history = []

    for i in range(self.epochs):

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
        y_pred = self.forward_prop(X_batch, training=True)

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
        self.backward_prop(y_batch, y_pred)
        self.optimizer.step(model=self, lr=self.alpha, epoch=i+1, X=X_train, y=y_train)
        
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
    
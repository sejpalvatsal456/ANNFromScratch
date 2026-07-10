# Artificial Neural Network (ANN) From Scratch

A fully connected Artificial Neural Network (ANN) implemented from scratch using **NumPy**/**CuPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch. This project demonstrates the complete training pipeline, including forward propagation, backpropagation, gradient descent, and prediction on handwritten digit images.

The network is trained on the MNIST Digit Recognizer dataset and is designed to be easily extendable with additional layers and activation functions.

## Features

### Version 0.1.0
* Forward propagation
* Backpropagation
* Gradient Descent optimization
* Multiple fully connected layers
* ReLU activation
* Softmax output layer
* Cross-Entropy loss
* One-hot encoding

### Version 0.1.1
* Prediction and accuracy evaluation
* Mini batch size training 
* GPU acceleration support

### Version 0.1.2
* Added Generalised template for optimizers
* Added Momentum optimizer
* Added learning rate decay parameter to models
---

### Version 0.1.3
* Added NAG, RMSProp and Adam Optimizer
* Redesigned the mechanism of optimzer

## Tech Stack

* Python
* NumPy
* CuPy
* Pandas
* Matplotlib

---

## Network Architecture

The network consists of a sequence of fully connected layers.

Example architecture:

```
Input (784)
      │
      ▼
Hidden Layer (128, ReLU)
      │
      ▼
Hidden Layer (64, ReLU)
      │
      ▼
Output Layer (10, Softmax)
```

The architecture can be changed simply by adding or removing layers.

---

## Documentation

### Creating a Model

Create a model by specifying the number of training iterations and learning rate.

```python
model = Model(
    iterations=100,
    alpha=0.2,
    batch_size=512,
    decay=0.001,
)
```

---

### Optimizers

You can use Optimizer class in `optimizer.py` file to create optimizer for the model.
Example -

```python
from optimizers import Optimzer

class MyOptimizer(Optimizer):
      def __init__(self, ...parameters):
            super.__init__(name="My Optimizer")
            # setting up parameters
      
      def step(self, model, grads, X, y, lr):
            # Here comes the logic to update the parameters of model according to gradients

```
**Note-** If `step()` is not implemented, calling it will give you `NotImplementedError`.

Then you can use the object of that subclass to use it.
For example, you can use the builtin `Momentum` Optimizer -

```python
from optimizers import Momentum

momentum = Momentum(beta=0.9)
model = Model(
      iterations=100,
      alpha=0.2,
      batch_size=512,
      decay=0.001,
      optimizer=momentum
)
```

**Note-** By default, model uses Mini Batch Gradient Descent.

---

### Creating Layers

A layer is created by specifying

* Number of neurons
* Number of input features
* Activation function
* Activation derivative (hidden layers only)

Example:

```python
l1 = Layer(128, 784, ReLu, ReLu_derive)
l2 = Layer(64, 128, ReLu, ReLu_derive)
l3 = Layer(10, 64, softmax)
```

The first hidden layer receives 784 inputs (28×28 image pixels) and produces 10 outputs. The output layer receives those 10 values and predicts probabilities for the 10 digit classes.

---

### Building the Network

Add layers in order.

```python
model.add_layer(l1)
model.add_layer(l2)
model.add_layer(l3)
```

### Training

Train the network using

```python
model.fit(X_train, y_train)
```

The model performs

* Forward propagation
* Loss gradient computation
* Backpropagation
* calls optimizer to update weights

for the specified number of iterations.

---

### Making Predictions

Predict labels for unseen data.

```python
predictions = model.predict(X_test)
```

---

### Evaluating Accuracy

```python
accuracy = model.accuracy(X_test, y_test)

print(accuracy)
```

---

### Visualizing Digits

Display any digit from the dataset.

```python
show_digit(X_train, y_train, index=0)
```

Or visualize model predictions.

```python
show_prediction(model, X_train, y_train, index=25)
```

---
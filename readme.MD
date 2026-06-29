# Artificial Neural Network (ANN) From Scratch

A fully connected Artificial Neural Network (ANN) implemented from scratch using **NumPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch. This project demonstrates the complete training pipeline, including forward propagation, backpropagation, gradient descent, and prediction on handwritten digit images.

The network is trained on the MNIST Digit Recognizer dataset and is designed to be easily extendable with additional layers and activation functions.

## Features

* Forward propagation
* Backpropagation using matrix calculus
* Gradient Descent optimization
* Multiple fully connected layers
* ReLU activation
* Softmax output layer
* Cross-Entropy loss
* One-hot encoding
* Prediction and accuracy evaluation
* Visualization of handwritten digits

---

## Tech Stack

* Python
* NumPy
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
Hidden Layer (10, ReLU)
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
    iterations=500,
    alpha=0.1
)
```

---

### Creating Layers

A layer is created by specifying

* Number of neurons
* Number of input features
* Activation function
* Activation derivative (hidden layers only)

Example:

```python
l1 = Layer(
    10,
    784,
    ReLu,
    ReLu_derive
)

l2 = Layer(
    10,
    10,
    softmax
)
```

The first hidden layer receives 784 inputs (28×28 image pixels) and produces 10 outputs. The output layer receives those 10 values and predicts probabilities for the 10 digit classes.

---

### Building the Network

Add layers in order.

```python
model.add_layer(l1)
model.add_layer(l2)
```

You can create deeper networks simply by adding more layers.

Example:

```python
model.add_layer(Layer(64, 784, ReLu, ReLu_derive))
model.add_layer(Layer(32, 64, ReLu, ReLu_derive))
model.add_layer(Layer(10, 32, softmax))
```

---

### Training

Train the network using

```python
model.fit(X_train, y_train)
```

The model performs

* Forward propagation
* Loss gradient computation
* Backpropagation
* Weight updates

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
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import Model, softmax, ReLu, ReLu_derive
from layers import Dense, Dropout
from backend import GPU, xp
from optimizers import Momentum, SGD, NAG, RMSProp, Adam

# df = pd.read_csv("./digit-recognizer/train.csv")
df = pd.read_csv("./fashion-mnist/fashion-mnist_train.csv")
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)


X_train = xp.asarray(X_train) / 255.0
X_test = xp.asarray(X_test) / 255.0
y_train = xp.asarray(y_train)
y_test = xp.asarray(y_test)

# optimizer = RMSProp(beta=0.9, epsilon=1e-8)
optimizer = Adam()

# model = Model(
#   100,
#   0.005,
#   batch_size=512,
#   optimizer=optimizer
# )
# l1 = Dense(128, 784, ReLu, ReLu_derive)
# l2 = Dropout(rate=0.1)
# l3 = Dense(64, 128, ReLu, ReLu_derive)
# l4 = Dropout(rate=0.1)
# l5 = Dense(10, 64, softmax)
# model.add_layer(l1)
# model.add_layer(l2)
# model.add_layer(l3)
# model.add_layer(l4)
# model.add_layer(l5)

layers = [
  Dense(128, ReLu, ReLu_derive),
  Dropout(rate=0.1),
  Dense(64, ReLu, ReLu_derive),
  Dropout(0.1),
  Dense(10, softmax)
]

model = Model(
  input_shape=(28, 28),
  layers=layers,
  epochs=100,
  alpha=0.005,
  batch_size=256,
  optimizer=optimizer
)

model.fit(X_train, y_train)

train_acc = model.accuracy(X_train, y_train)
print(f"Train Accuracy: {train_acc:.4f}")

test_acc = model.accuracy(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ANN import Model, Layer, softmax, ReLu, ReLu_derive
from backend import GPU, xp

df = pd.read_csv("./digit-recognizer/train.csv")
X = df.drop(columns='label')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)
X_train = xp.asarray(X_train) / 255.0
X_test = xp.asarray(X_test) / 255.0
y_train = xp.asarray(y_train)
y_test = xp.asarray(y_test)

model = Model(
  100,
  0.2,
  batch_size=512
)
l1 = Layer(128, 784, ReLu, ReLu_derive)
l2 = Layer(64, 128, ReLu, ReLu_derive)
l3 = Layer(10, 64, softmax)
model.add_layer(l1)
model.add_layer(l2)
model.add_layer(l3)
model.fit(X_train, y_train)

train_acc = model.accuracy(X_train, y_train)
print(f"Train Accuracy: {train_acc:.4f}")

test_acc = model.accuracy(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
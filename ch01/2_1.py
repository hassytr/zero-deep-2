import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

W1 = np.random.randn(2,4)
b1 = np.random.randn(4)
W2 = np.random.randn(4,3)
b2 = np.random.randn(3)
x = np.random.randn(10,2)

h = np.dot(x, W1) + b1
a = sigmoid(h)
s = np.dot(a, W2) + b2

print(s)
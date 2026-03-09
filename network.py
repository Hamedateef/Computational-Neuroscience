import numpy as np

def tanh(x):
    return np.tanh(x)

i1 = 0.05
i2 = 0.10
X = np.array([i1, i2])

W1 = np.random.uniform(-0.5, 0.5, (2, 2))
W2 = np.random.uniform(-0.5, 0.5, (2, 2))

b1 = 0.5
b2 = 0.7

hidden_input = np.dot(W1, X) + b1
hidden_output = tanh(hidden_input)

final_input = np.dot(W2, hidden_output) + b2
final_output = tanh(final_input)

print("Network Output:", final_output)
import numpy as np

np.random.seed(1)

# ---------- Data (XOR truth table) ----------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)          # shape (4, 2)
y = np.array([[0], [1], [1], [0]], dtype=float)   # shape (4, 1)

# ---------- Initialize weights ----------
W1 = np.random.randn(2, 4)      # input -> hidden
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)      # hidden -> output
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 10

#Training
for epoch in range(epochs):

    # Forward pass
    z1 = X @ W1 + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))

    # Loss (mean squared error
    loss = np.mean((a2 - y) ** 2)

    # Backward pass (chain rule)
    dloss_da2 = 2 * (a2 - y) / len(X)
    dz2 = dloss_da2 * a2 * (1 - a2)
    dW2 = a1.T @ dz2                          
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1 - a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


    print(f"epoch {epoch:5d}  loss = {loss:.5f}")

print(f"final loss = {loss:.5f}")
print()

#Test
z1 = X @ W1 + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = a1 @ W2 + b2
pred = 1 / (1 + np.exp(-z2))

print("x1  x2  target  prediction  rounded")
for i in range(4):
    print(f"{int(X[i,0])}   {int(X[i,1])}   {int(y[i,0])}       {pred[i,0]:.4f}      {int(round(pred[i,0]))}")
import numpy as np

np.random.seed(0)
x = np.random.randn(8, 4) * 5 + 3
N, D = x.shape

eps = 1e-5
gamma = np.ones(D)
beta = np.zeros(D)

print("BEFORE normalization (x):")
print(x.round(2))
print()
print("mean per row:", x.mean(axis=1).round(2))
print("std per row: ", x.std(axis=1).round(2))
print()

# normalize
mean = x.mean(axis=1, keepdims=True)
var = x.var(axis=1, keepdims=True)
x_hat = (x - mean) / np.sqrt(var + eps)
out = gamma * x_hat + beta

print("AFTER normalization:")
print(out.round(2))
print()
print("mean per row:", out.mean(axis=1).round(2))
print("std per row: ", out.std(axis=1).round(2))
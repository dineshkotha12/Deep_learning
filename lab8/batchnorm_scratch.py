import numpy as np

np.random.seed(0)
x = np.random.randn(8, 4) * 5 + 3      # 8 samples, 4 features
N, D = x.shape

eps = 1e-5
gamma = np.ones(D)
beta = np.zeros(D)

print("BEFORE normalization (x):")
print(x.round(2))
print()
print("mean per column:", x.mean(axis=0).round(2))
print("std per column: ", x.std(axis=0).round(2))
print()

# normalize
mean = x.mean(axis=0)
var = x.var(axis=0)
x_hat = (x - mean) / np.sqrt(var + eps)
out = gamma * x_hat + beta

print("AFTER normalization:")
print(out.round(2))
print()
print("mean per column:", out.mean(axis=0).round(2))
print("std per column: ", out.std(axis=0).round(2))
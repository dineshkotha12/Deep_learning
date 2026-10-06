import torch
import torch.nn as nn

# Define the network
model = nn.Sequential(
    nn.Linear(3, 5),    # input (3) -> hidden layer 1 (5 neurons)
    nn.ReLU(),
    nn.Linear(5, 5),    # hidden layer 1 -> hidden layer 2 (5 neurons)
    nn.ReLU(),
    nn.Linear(5, 1)     # hidden layer 2 -> output (1 value)
)

print(model)
print()

# Test with a sample input
x = torch.tensor([[1.0, 2.0, 3.0]])     # 1 sample, 3 values
y = model(x)

print("input: ", x)
print("output:", y)

# Output with 2 neurons of hidden layer 1 dropped
h1 = model[1](model[0](x))              # hidden layer 1 output, shape (1, 5)
print("hidden layer 1 before dropout:", h1.detach().numpy().round(3))

mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])   # neurons 2 and 4 are OFF
h1_dropped = h1 * mask
print("mask:                         ", mask.numpy())
print("hidden layer 1 after dropout: ", h1_dropped.detach().numpy().round(3))

h2 = model[3](model[2](h1_dropped))     # hidden layer 2
y_dropped = model[4](h2)                # output
print()
print("output (2 neurons dropped):", y_dropped.item())

#Same, but scaled like inverted dropout
h1_scaled = h1 * mask / (1 - 2/5)       # multiply the 3 survivors by 5/3
h2 = model[3](model[2](h1_scaled))
y_scaled = model[4](h2)
print("output (dropped + scaled): ", y_scaled.item())




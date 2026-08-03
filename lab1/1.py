import os
os.environ["http"]="http://dineshkotha3%40gmail.com:dinesh%402802@proxy.ibab.ac.in:3128"
os.environ["https"]="https://dineshkotha3%40gmail.com:dinesh%402802@proxy.ibab.ac.in:3128"
import numpy as np
import matplotlib.pyplot as plt

z_values = np.linspace(-10, 10, 100)

print(z_values)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z)/((1+np.exp(-z))*(1+np.exp(-z)))

def tanh_derivative(z):
    return (np.exp(z)*np.exp(z))*((-np.exp(-z))*(-np.exp(-z)))*(np.exp(z)-np.exp(-z))*(np.exp(z)-np.exp(-z))/ ((np.exp(z)+np.exp(-z))*(np.exp(z)+np.exp(-z)))

def relu(z):
    if z<0:
        return 0
    else:
        return z

def leaky_relu(z):
    w=0.1
    if z>0:
        return z
    else:
        return w*z


sigmoid_output = sigmoid(z_values)
sigmoid_d= sigmoid_derivative(z_values)

tanh_output = tanh(z_values)
tanh_d = tanh_derivative(z_values)

relu_output = []

for value in z_values:
    relu_output.append(relu(value))

print(relu_output)

leaky_relu_output = []

for value in z_values:
    leaky_relu_output.append(leaky_relu(value))

print("leaky_relu=",leaky_relu_output)

print("sigmoid_output=",sigmoid_output)
print("sigmoid_derivative=",sigmoid_d)

print("tanh_output=",tanh_output)
print("tanh_derivatives=", tanh_d)

print("relu_output",relu_output)

plt.plot(z_values, sigmoid_output)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.show()

plt.plot(z_values, sigmoid_d)
plt.title("Sigmoid derivative Function")
plt.xlabel("z")
plt.ylabel("Sigmoid_derivative(z)")
plt.grid(True)
plt.show()

plt.plot(z_values, tanh_output)
plt.title("tanh Function")
plt.xlabel("z")
plt.ylabel("tanh(z)")
plt.grid(True)
plt.show()

plt.plot(z_values, tanh_d)
plt.title("tanh Function")
plt.xlabel("z")
plt.ylabel("tanh(z)")
plt.grid(True)
plt.show()

plt.plot(z_values, relu_output)
plt.title("relu Function")
plt.xlabel("z")
plt.ylabel("relu(z)")
plt.grid(True)
plt.show()


import numpy as np
#3 layered



x=[6,4,7,3]


# Hidden Layer 1
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3)

# Hidden Layer 2
W2 = np.random.randn(2, 3)
b2 = np.random.randn(2)

# Output Layer
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1)

def relu(z):
    if z<0:
        return 0
    else:
        return z

# Hidden Layer 1
z1 = np.dot(W1,x) + b1
a1=[]
for i in z1:
    a=relu(i)
    a1.append(a)

print(z1)
print("layer1=",a1)

#hidden layer2
z2=np.dot(W2,a1)+b2
a2=[]
for i in z2:
    a=relu(i)
    a2.append(a)

print(z2)
print("layer2=",a2)

#output layer
z3=np.dot(W3,a2)+b3
y=relu(z3)

print("y=",y)




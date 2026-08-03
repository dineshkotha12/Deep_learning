import numpy as np

#single layer

x=[1,6,3,5]
w=[0.3,0.1,0.09,0.2]

#single layer

b=0.02
z=np.dot(x,w)+b

print(z)

def relu(z):
    if z<0:
        return 0
    else:
        return z

y= relu(z)

print("activation function=",y)


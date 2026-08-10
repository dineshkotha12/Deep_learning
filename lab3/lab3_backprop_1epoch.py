import numpy as np

#single layer

x=[1,6,3,5]
w=[0.3,0.1,0.09,0.2]

y_actual=3

#single layer

b=0.02
z=np.dot(x,w)+b

print(z)

def relu(z):
    if z<0:
        return 0
    else:
        return z

def loss(y,y_actual):
    return (1/2)*(y-y_actual)*(y-y_actual)

y= relu(z)
loss= loss(y,y_actual)

print("activation function=",y)
print("loss=",loss)

#backprop

dL_dy = y - y_actual
print("dL/dy =", dL_dy)
y = relu(z)

def relu_derivative(z):
    if z > 0:
        return 1
    else:
        return 0


dy_dz = relu_derivative(z)
print("dy/dz =", dy_dz)

dL_dz = dL_dy * dy_dz
print("dL/dz =", dL_dz)

z = np.dot(x, w) + b

dL_dw = dL_dz * np.array(x)
print("dL/dw =", dL_dw)

dL_db = dL_dz
print("dL/db =", dL_db)

learning_rate = 0.01
w = np.array(w)
w = w - learning_rate * dL_dw
print("updated w =", w)

b = b - learning_rate * dL_db
print("updated b =", b)



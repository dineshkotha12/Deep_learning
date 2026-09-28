from email.parser import BytesParser

import numpy as np
from scipy.special.cython_special import hyp0f1

#declaring weights

w_xh = np.array([0.1,0.2])
w_hh = np.array([0.01,0.1])
w_hy = np.array([0.12,0.32])

ho= np.array([0,0])
xt= 5

for i in range(5):

    ht= np.tanh((w_hh*ho)+(w_xh*xt))
    yt=ht*w_hy
    print(ht)
    ht = ho
    xt=yt


print(ht)
print(yt)


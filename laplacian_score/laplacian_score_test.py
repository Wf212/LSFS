#/usr/bin/python
# -*- coding=utf-8 -*-


import pandas as pd
import numpy as np
import scipy as sp
import sys


def test_filter_KNN():
    a = np.random.randint(10000,size=(120,120))
    b = filter_KNN(a, 1)
    err = False
    for i,bb in enumerate(b):
#         print(type(bb), bb.shape, bb[bb!=0])
        if np.sum(bb!=0) > 0 and np.min(bb[bb!=0]) != np.min(a[i,:]):
#             print(a[i, :], bb)
            err = True
    if err == True:
        print("error")
    else:
        print("not found error")






#/usr/bin/python
# -*- coding=utf-8 -*-



import pandas as pd
import numpy as np
import scipy as sp
import sys


def append_module_path():
    import sys
    paths = [ \
        "../gen_data",
        "../evaluate",
        "../read_data",
        "../PRPC"
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
        
append_module_path()
import gen_data
import evaluate
import read_data
import PRPC_FUN


from laplacian_score_FUN import *
from general_algorithm import *

x_train, x_test, y_train, y_test = gen_data.gen_data()
S = compute_S(x_train)
S_new = filter_KNN(S, k=100)
D = compute_D(S_new)
L = compute_L(D, S_new)
feature_score = compute_laplacian_score(x_train, L, D)

print("feature score=>\n", feature_score)



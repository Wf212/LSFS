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


def compute_S(x_train):
    """
    特征
    """
    f_len = x_train.shape[1]
    x_f = set(range(f_len))

    """
    计算S矩阵
    """
    t = 10
    n_sampels = x_train.shape[0]
    S = np.zeros((n_sampels, n_sampels))
    for i in range(n_sampels):
        for j in range(n_sampels): 
            value = - np.linalg.norm(x_train[i] - x_train[j], ord=2) ** 2 / t
            S[i, j] = np.e **  value
    return S


def compute_D(S):
    """
    计算D矩阵
    """
    n_sampels = S.shape[0]
    D = np.diag(np.dot( S, np.ones((n_sampels, 1)).reshape(-1) ))
    return D


def compute_L(D, S):
    """
    计算L矩阵
    """
    L = D - S
    return L



def compute_laplacian_score(x_train, L, D):
    """
    计算特征的分数
    """
        
    """
    特征
    """
    f_len = x_train.shape[1]
    x_f = set(range(f_len))
    
    n_sampels = x_train.shape[0]
    
    feature_laplacian_score = np.zeros(f_len)
    for f_i in x_f:
        """
        第i个特征
        """
        x_fi = x_train[:,f_i:f_i+1]
        
        a = np.dot( np.dot( x_fi.T, D ), np.ones((n_sampels, 1)) )
        b = np.dot( np.dot( np.ones((1, n_sampels)), D ), np.ones((n_sampels, 1)) )
        x_fi_new = x_fi - a / b * np.ones((n_sampels, 1))
        
        c = np.dot( np.dot( x_fi_new.T, L ), x_fi_new )
        d = np.dot( np.dot( x_fi_new.T, D ), x_fi_new )
        L_i =  c / d
            
#         print(L_i)
        
        feature_laplacian_score[f_i] = L_i
    
    return feature_laplacian_score



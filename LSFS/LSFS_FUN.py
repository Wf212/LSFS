#!usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time
import sys
from LSFS_TEST import print_W

from EProjSimplex_new import *


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


def norm_2_1(a):
    """
    对每行的向量求第二范数，然后对所有范数求和
    """
    return np.sum(np.linalg.norm(a, ord = 2, axis=1))


def fun22_value(W, X, H, Q, Y, gama = 10**-6):
    """
    ||H*X.T*W - H*Y||F范数 ^ 2 + gama * (W.T*Q*W)的迹
    """
#    gama = 10**-6
    return np.linalg.norm(np.dot(np.dot(H, X.T), W) - np.dot(H, Y), ord = "fro")**2 + gama*np.trace(np.dot(np.dot(W.T, Q),W))


def fun17_value(W, X, Y, gama = 10**-6):
    """
    ||H*X.T*W - H*Y||F范数 ^ 2 + gama * (W.T*Q*W)的迹
    """
#    gama = 10**-6
#     print(X.shape, Y.shape, W.shape)
    b = compute_b(X, Y, W)
#     print("17 value", np.linalg.norm(np.dot(X.T, W) + np.dot(np.ones((X.shape[1],1)), b.T) - Y, ord = "fro")**2 + gama*(norm_2_1(W)**2))
    return np.linalg.norm(np.dot(X.T, W) + np.dot(np.ones((X.shape[1],1)), b.T) - Y, ord = "fro")**2 + gama*(norm_2_1(W)**2)


def fun8_value(X, Y, W, b, gama = 10**-6):
    """
    X : d x n
    
    ||X.T * W + 1*b.T - Y||的L2范数 ^ 2 + gama * ( ||W||的F范数 ^ 2 )
    """
#    gama = 10**-6
    n = X.shape[1]
    return np.linalg.norm( np.dot(X.T,W) + np.dot(np.ones((n, 1)),b.T) - Y , ord=2)**2 + gama*(np.linalg.norm( W , ord = "fro")**2)


def compute_W(X, Y, H, Q, gama = 10**-6):
#     gama = 10**-6
#    gama = 60
    """
    W = (X*H*X.T + gama * Q)^-1 * X*H*Y
    """
    W = np.dot( np.dot( np.dot( \
                np.linalg.inv( np.dot( np.dot(X,H), X.T)+gama*Q ) \
                               , X), H), Y)
    return W



def compute_H(n):
    """
    I => n x n
    1 => n x 1
    H = I - 1/n * 1 * 1.T
    """
    H = np.eye(n,n) - 1/n*np.ones((n,n))
    return H



def compute_Q(W):
    """
    q(ij) = ||W||2,1 / ||w^j||2
    
    axis = 1 =》 对W的每一行求L2范数
    np.linalg.norm(W, ord=2, axis = 1) =》 对每行求L2范数
    """
    Q = norm_2_1(W) / np.linalg.norm(W, ord = 2, axis=1)
    Q = np.diag(Q)
    return Q



def get_W(X, Y, gama = 10**-6):
    
    """
    d特征，c类别，n样本数
    
    X : (d x n)
    Y : (n x c)
    
    算法中使用的X的维度是(d x n)
    """
    
    d, n = X.shape
    c = Y.shape[1]
    
    # Q初始化为一个单位矩阵
    Q = np.eye(d)
    
#     print(Q)
#     print("====================")
    
    # H矩阵不变，算一遍即可
    H = compute_H(n)
    
    W = compute_W(X, Y, H, Q, gama = gama )
    Q = compute_Q(W)
#    pre_f = cur_f = fun22_value(W, X, H, Q, Y, gama = gama)
    pre_f = cur_f = fun17_value(W, X, Y, gama = gama)
    
    # nan 判断
    if cur_f != cur_f:
        print("nei nan")
        return W
    
#     print(W)
#     print()
#     print(Q)
#     print("====================")
    
    NITER = 30
    epsilon = 10**-8
    for i in range(NITER):
        pre_f = cur_f
        W = compute_W(X, Y, H, Q, gama = gama)
        Q = compute_Q(W)
        
#         print_W(W)
        
#        cur_f = fun22_value(W, X, H, Q, Y, gama = gama)
        cur_f = fun17_value(W, X, Y, gama = gama)
        
        # nan 判断
        if cur_f != cur_f:
            print("nei nan")
            break
            
        # coverage
        if cur_f == 0 and abs((cur_f - pre_f)/(cur_f + epsilon)) < epsilon:
            break
            
        if cur_f != 0 and abs((cur_f - pre_f)/(cur_f)) < epsilon:
            break
    return W


def compute_YU(X, W, b):
    """
    X : (d x n)
    """
    c = W.shape[1]
    YU = np.zeros((X.shape[1], c))
    # 对于每一个样本，维度 1 x d
    for i in range(X.shape[1]):
        """
        min ( ||(xi.T) * W + b.T - yi.T||的F范数 ^ 2 )
        s.t. yi>=0, 1*yi=1
        """
        ad = np.dot(X[:,i:i+1].T, W) + b.T
        ad_new, ft = EProjSimplex_new(ad)
        YU[i:i+1,:] = ad_new.A
    return YU



def compute_b(X, Y, W):
    """
    X : d x n
    Y : n x c
    W : d x c
    
    b = 1/n * (Y.T * 1 - W.T * X * 1)
    1 是 n x 1 维的全1矩阵
    """
    n = X.shape[1]
    b = 1/n*(np.dot(Y.T, np.ones((n,1))) - np.dot(np.dot(W.T, X), np.ones((n,1))))
    return b



def get_new_X_Y_YU_W_f(X, Y, XL, YL, XU, gama = 10**-6):
    """
    X : d x n
    Y : n x c
    XL : nl x d
    YL : nl x c
    XU : nu x d
    """
#     n = X.shape[1]
    W = get_W(X, Y, gama = gama)
    
#     print_W(W)
#     b = 1/n*(np.dot(Y.T, np.ones((n,1))) - np.dot(np.dot(W.T, X), np.ones((n,1))))
    b = compute_b(X, Y, W)
    
    YU = compute_YU(XU.T, W, b)
    
    X = sp.concatenate((XL, XU), axis = 0)
    Y = sp.concatenate((YL, YU), axis = 0)
    
    X = X.T
    cur_f = fun8_value(X, Y, W, b, gama = gama)
    return X, Y, YU, W, cur_f




def compute_thea(W):
    """
    W : d x c
    thea_j = ||w_j||2 / sum(||w_j||2) 
                        j=1:d 
    """
    # 对W的每行求L2范数，再求和
    W_L2_sum = np.sum(np.linalg.norm(W, ord=2, axis = 1))
    # 对W的每行求L2范数
    s = np.linalg.norm(W, ord=2, axis = 1) / W_L2_sum
    return s



def compute_acc_diff_gama_fearture(XL_train, YL_train, XU_train, YU_train, gama_array, idx_array\
                                   , output_file_name="feature_order"):
    XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
    YL = read_data.label_n1_to_nc(YL)
    YU = read_data.label_n1_to_nc(YU)
    data = []
    for gama in gama_array:
        print("gama : ", gama)
        feature_order, time_dual = lsfs( XL, YL, XU, output_file_name=output_file_name, gama = gama )
        acc_array = evaluate.cal_many_acc_by_idx(XL_train, YL_train, XU_train, YU_train,\
                               feature_order, idx_array)
        data.append(acc_array)
    return np.array(data)


            
def lsfs(XL, YL, XU, output_file_name="feature_order", gama = 10**-6):
    start_time = time.clock()

    X, Y, YU, W, cur_f = get_new_X_Y_YU_W_f(XL.T, YL, XL, YL, XU, gama = gama)
    # nan 判断  
    if cur_f != cur_f:
        s = compute_thea(W)
        feature_order = list( np.argsort(s) )
        feature_order = feature_order[::-1]
        print("wai nan")
        return feature_order, None
    
    print_W(W)

    NITER = 20
    epsilon = 10**-8
    for i in range(NITER):
        pre_f = cur_f

        X, Y, YU, W, cur_f = get_new_X_Y_YU_W_f(X, Y, XL, YL, XU, gama = gama)

        
        print_W(W)

#         print(cur_f)
        
    
        # nan 判断  
        if cur_f != cur_f:
            break
        
        # coverage 
        if cur_f == 0 and abs((cur_f - pre_f)/(cur_f + epsilon)) < epsilon:
            break
            
        if cur_f != 0 and abs((cur_f - pre_f)/(cur_f)) < epsilon:
            break


    s = compute_thea(W)
    feature_order = list( np.argsort(s) )
    feature_order = feature_order[::-1]

    time_dual = time.clock() - start_time
    
    with open(output_file_name, "w+") as result_file:
        print("\n".join([str(w) for w in feature_order]), file=result_file)
    
    return feature_order, time_dual
            
            


#/usr/bin/python
# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
import sys
import time


def append_module_path():
    import sys
    paths = [ \
        "../gen_data",
        "../evaluate",
        "../read_data",
        "../PRPC",
        "../LSDF"
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
        
append_module_path()
import gen_data
import evaluate
import read_data
import PRPC_FUN
import LSDF_FUN
from general_algorithm import *


def compute_S(x_train):
    t = 100
    """
    特征
    """
    f_len = x_train.shape[1]
    x_f = set(range(f_len))
    
    """
    计算S矩阵
    """
#     t = 100
    n_sampels = x_train.shape[0]
    S = np.zeros((n_sampels, n_sampels))
    for i in range(n_sampels):
        for j in range(n_sampels): 
            value = np.linalg.norm(x_train[i] - x_train[j], ord=2) ** 2 / t
            S[i, j] = np.e **  -value
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
        
        
        # 求取新的fr
        a = np.dot( np.dot( x_fi.T, D ), np.ones((n_sampels, 1)) )
        b = np.dot( np.dot( np.ones((1, n_sampels)), D ), np.ones((n_sampels, 1)) )
        x_fi_new = x_fi - a / b * np.ones((n_sampels, 1))
        
        # 求L_i
        c = np.dot( np.dot( x_fi_new.T, L ), x_fi_new )
        d = np.dot( np.dot( x_fi_new.T, D ), x_fi_new )
        L_i =  c / d
            
#         print(L_i)
        
        feature_laplacian_score[f_i] = L_i
    
    return feature_laplacian_score


def feature_lab_score(x_train, k= 10):
    S = compute_S(x_train)
    # 将S中非k近邻的S值设置为0
    distance_m = LSDF_FUN.get_distance_array(x_train)
    knn_flag_m = get_KNN_flag(distance_m, k=k)
    for i in range( len(knn_flag_m) ):
        knn_flag_m[i, i] = True
    knn_flag_m = knn_flag_m.T + knn_flag_m
    S_new = S.copy()
    S_new[~knn_flag_m] = 0 
    D = compute_D(S_new)
    L = compute_L(D, S_new)
    feature_score = compute_laplacian_score(x_train, L, D)
    return feature_score


def feature_lab_score2(x_train, k = 10):
    S = compute_S(x_train)
    # 将S中非k近邻的S值设置为0
    distance_m = LSDF_FUN.get_distance_array(x_train)
    knn_flag_m = get_KNN_flag(distance_m, k=k)
    for i in range( len(knn_flag_m) ):
        knn_flag_m[i, i] = True
    knn_flag_m = knn_flag_m.T + knn_flag_m
    S_new = S.copy()
    S_new[~knn_flag_m] = 0 
    
    X = x_train.copy()
    W = S_new

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    D = np.diag(D)
    tmp = np.dot(D, X)
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D))
    t2 = np.transpose(np.dot(Xt, L))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000

    # compute laplacian score for all features
    score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
    return np.transpose(score)


def laplacian_score_feature_order(x_train, k = 10, output_file_name="feature_order"):
    start_time = time.clock()
    feature_score = feature_lab_score(x_train, k = k)
    time_dual = time.clock() - start_time

    feature_order = list( np.argsort(feature_score) )
    feature_order = feature_order[::-1]
    with open(output_file_name, "w+") as result_file:
        print("\n".join([str(w) for w in feature_order]), file=result_file)
    return feature_order, time_dual


def laplacian_score_feature_order2(x_train, k = 10, output_file_name="feature_order"):
    start_time = time.clock()
    feature_score = feature_lab_score2(x_train, k = k)
    time_dual = time.clock() - start_time

    feature_order = list( np.argsort(feature_score) )
    feature_order = feature_order[::-1]
    with open(output_file_name, "w+") as result_file:
        print("\n".join([str(w) for w in feature_order]), file=result_file)
    return feature_order, time_dual

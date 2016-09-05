#!usr/bin/python
# -*- coding:utf-8 -*-

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
        "../laplacian_score"
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
        
append_module_path()
import general_algorithm

def get_distance_array(x_train):
    """
    求距离矩阵
    """
    
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
            value = np.linalg.norm(x_train[i] - x_train[j], ord=2)
            S[i, j] = value
    return S




def get_bool_matrix(data):
    """
    求bool矩阵
    不等于0的用True
    其余用False
    """
    m, n = data.shape
    data_new = np.zeros((m, n), dtype=np.bool)
    for i in range( m ):
        for j in range( n ):
            if data[i, j] != data.dtype.type(0):
                a_new[i, j] = 1
    return data_new



def compute_Sw(x_labeled, y_labeled, x_unlabeled):
    """
    对输出Sw矩阵做了约定：
        假设ln个带标签的样本
        假设un个未带标签的样本
        那么矩阵索引0~ln-1对应于带标签的样本，ln~ln+un-1对应于未带标签的样本
    """
    gama  = 100
    k = 5
    ln = x_labeled.shape[0]
    un = x_unlabeled.shape[0]
    x = np.concatenate((x_labeled, x_unlabeled), axis = 0)
    n_samples = x.shape[0]
    S = get_distance_array(x)
    """
    filter_KNN包括了自身，所以要k+1
    """
    s_knn = general_algorithm.get_KNN_flag(S, k=k+1)
    Sw = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if (i< ln and j < ln) and y_labeled[i] == y_labeled[j]:
                Sw[i, j] = gama
            elif (i>= ln or j >= ln) and s_knn[i, j] == True \
                or s_knn[j, i] == True:
                Sw[i, j] = 1
            else:
                Sw[i, j] = 0
    return Sw



def compute_Sb(x_labeled, y_labeled, x_unlabeled):
    """
    对输出Sb矩阵做了约定：
        假设ln个带标签的样本
        假设un个未带标签的样本
        那么矩阵索引0~ln-1对应于带标签的样本，ln~ln+un-1对应于未带标签的样本
    """
    ln = len(x_labeled)
    ln = x_labeled.shape[0]
    un = x_unlabeled.shape[0]
    n_samples = ln + un
    Sb = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if (i < ln and j < ln) and y_labeled[i] != y_labeled[j] :
                Sb[i, j] = 1
    return Sb



def compute_Lw(Sw):
    n_samples = Sw.shape[0]
    Dw = np.diag( np.dot(Sw, np.ones((n_samples, 1))).reshape(-1) )
    Lw = Dw - Sw
    return Lw



def compute_Lb(Sb):
    n_samples = Sb.shape[0]
    Db = np.diag( np.dot(Sb, np.ones((n_samples, 1))).reshape(-1) )
    Lb = Db - Sb
    return Lb




def compute_Lr(x_labeled, x_unlabeled, Lw, Lb):
    f_n = x_labeled.shape[1]
    Lr = np.zeros(f_n)
    x = np.concatenate((x_labeled, x_unlabeled))
    for f_i in range(f_n):
        x_fi = x[:, f_i:f_i+1]
        a = np.dot( np.dot( x_fi.T, Lb ), x_fi)
        b = np.dot( np.dot( x_fi.T, Lw ), x_fi)
        Lr[f_i] = a / b
    return Lr




def lsdf(x_labeled, y_labeled, x_unlabeled, output_file_name="feature_order"):
    start_time = time.clock()
    
    f_n = x_labeled.shape[1]
    Sw = compute_Sw(x_labeled, y_labeled, x_unlabeled)
    Sb = compute_Sb(x_labeled, y_labeled, x_unlabeled)
    Lw = compute_Lw(Sw)
    Lb = compute_Lb(Sb)
    Lr = compute_Lr(x_labeled, x_unlabeled, Lw, Lb)
    
    feature_order = np.argsort(Lr)
    feature_order = feature_order[::-1]
    
    time_dual = time.clock() - start_time
    
    with open(output_file_name, "w+") as result_file:
        print("\n".join([str(w) for w in feature_order]), file=result_file)
    
    return feature_order, time_dual







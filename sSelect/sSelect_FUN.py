import pandas as pd
import numpy as np
import scipy as sp
import sys
import math
import time


from normalized_mutual_info_score import *


def append_module_path():
    import sys
    paths = [ \
        "../gen_data",
        "../evaluate",
        "../read_data",
        "../PRPC",
        "../LSDF",
        "../laplacian_score"
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
import laplacian_score_FUN



def compute_gaussion_similarity(xi, xj, theta = 1):
    """
    传入两个一维向量，
    求这两个向量的高斯相似度
    """
    xi = np.array(xi)
    xj = np.array(xj)
    xi = xi.reshape(-1)
    xj = xj.reshape(-1)
    sub_square_sum = np.sum((xi - xj) ** 2)
#     print(sub_square_sum)
    similarty = np.e ** ( - sub_square_sum / (2 * (theta ** 2 ) ) )
    return similarty


def compute_W_by_distM(distance, theta = 1):
    """
    传入距离矩阵，
    转化为高斯相似度
    """
    similarty = np.e ** ( - distance * distance / (2 * (theta ** 2 ) ) )
    return similarty





def compute_W(X):
    X = x_train.copy()
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        xi = X[i:i+1, :]
        for j in range(n_samples):
            xj = X[j:j+1, :]
            W[i, j] = compute_gaussion_similarity(xi, xj, theta = 10)
    return W



def compute_S(X, W, d):
    X = X.copy()
    n_samples = X.shape[0]
    f_n = X.shape[1]
    S = np.zeros((n_samples, f_n))
    volv = np.sum(W)
    for f_i in range(f_n):
        x_fi = X[:, f_i]
#         print(x_fi.shape)
        a = 0
        for j in range(n_samples):
            a += x_fi[j]*d[j]
        gi = x_fi - a / volv * np.ones(n_samples)
        S[:, f_i] = gi
    return S



def select_Sg(S):
    fn = S.shape[1]
    select_g = []
    for i in range(fn):
        a = np.sum( S[:,i] * d )
        if a < 1 ** -9:
            select_g.append(i)
    return select_g



def compute_first_term(S, W, d, g_index, namuda = 0.1):
    n_samples = S.shape[0]
    numerator = 0.0
    denominator = 0.0
    for i in range(n_samples):
        gi = S[i, g_index]
        denominator += 2 * (gi ** 2) * d[i]
        for j in range(n_samples):
            gj = S[j, g_index]
            numerator +=  ( (gi - gj) ** 2 ) * W[i, j]
            
    return namuda * numerator / denominator



def compute_second_term(S, YL,g_index, namuda = 0.1):
    """
    做一个约定，
    S矩阵的前面部分全部对应带标签的，
    后部分对应不带标签的
    """

    import sklearn as sk
    import time
    from sklearn import cluster

    label_true = YL.reshape(-1).copy()
    n_clusters = len( set(label_true) )
    ln = len(label_true)

    data = S[:ln, g_index:g_index+1]
    
    kmeans_model = cluster.KMeans(n_clusters=n_clusters, random_state=int( time.clock() ))
    label_pred = kmeans_model.fit_predict(data)
    nmi_value = normalized_mutual_info_score2(label_true, label_pred)
#     print(nmi_value)
#     return nmi_value
    return (1.0-namuda) * ( 1.0 - nmi_value)




def compute_Lr(XL, YL, XU, k = 10, theta = 10, namuda = 0.1):
    """
    X的
    前面对应带标签数据
    后面对应不带标签数据
    """
    
    X = np.concatenate((XL, XU), axis = 0)
    YL = YL.copy()
    start_time = time.clock()
    
    distance_array = LSDF_FUN.get_distance_array(X)
    knn_flag_array = laplacian_score_FUN.get_KNN_flag(distance_array, k=k)
    
    W = compute_W_by_distM(distance_array, theta=theta)
    knn_flag_array = knn_flag_array + knn_flag_array.T
    # 非k近邻元素设置为0
    W[~knn_flag_array] = 0
    
    # 求每行的权值的和
    d = np.sum(W, axis=1)
    
    S = compute_S(X, W, d)
    
    
#     print(W.shape, d.shape)
    
    gn = S.shape[1]
    Lr = np.zeros(gn)
    for g_index in range(gn):
        first_term = compute_first_term(S, W, d, g_index, namuda= namuda) 
        second_term = compute_second_term(S, YL, g_index, namuda= namuda)
#         print(first_term, second_term)
        Lr[g_index] = first_term + second_term
    return Lr



def sSelect(XL, YL, XU, k = 10, theta = 10, namuda = 0.1, output_file_name="feature_order"):
    
    start_time = time.clock()
    
    f_score = compute_Lr(XL, YL, XU, k = 10, theta = 10, namuda = 0.1)
    feature_order = np.argsort(f_score)
#     feature_order = feature_order[::-1]
    
    
    time_dual = time.clock() - start_time
    
    with open(output_file_name, "w+") as result_file:
        print("\n".join([str(w) for w in feature_order]), file=result_file)
    
    return feature_order, time_dual




import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time

def norm_2_1(a):
    """
    对每行的向量求第二范数，然后对所有范数求和
    """
    return np.sum(np.linalg.norm(a,axis=1))


def fun22_value(W, X, H, Q, Y):
    """
    ||H*X.T*W - H*W||F范数 ^ 2 + gama * (W.T*Q*W)的迹
    """
    gama = 10^-6
    return np.linalg.norm(np.dot(np.dot(H, X.T), W) - np.dot(H, Y))**2 + gama*np.trace(np.dot(np.dot(W.T, Q),W))


def fun8_value(X, Y, W, b):
    """
    X : d x n
    
    ||X.T * W + 1*b.T - Y||的L2范数 ^ 2 + gama * ( ||W||的F范数 ^ 2 )
    """
    gama = 10^-6
    n = X.shape[1]
    return np.linalg.norm( np.dot(X.T,W) + np.dot(np.ones((n, 1)),b.T) - Y , ord=2)**2 + gama*(np.linalg.norm( W )**2)



"""
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
"""
def EProjSimplex_new(v, k=1):    
    v = np.matrix(v)
    ft = 1;
    n = np.max(v.shape)
    
#    if len(v.A[0]) == 0:
#        return v, ft
 
    if np.min(v.shape) == 0:
        return v, ft
    
#    print('n : ', n)
#    print(v.shape)
#    print('v :  ', v)
    
    v0 = v - np.mean(v) + k/n
    
#    print('v0 :  ', v0)
    
    vmin = np.min(v0)
    
    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10**-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if ft > 100:
                v1[v1<0] = 0.0
                break
        x = v1.copy()
        x[x<0] = 0.0
    else:
        x = v0
    return x,ft


def get_W(X, Y):
    
    """
    d特征，c类别，n样本数
    
    X : (d x n)
    Y : (n x c)
    
    算法中使用的X的维度是(d x n)
    """
    
    d, n = X.shape
    c = Y.shape[1]
    
    Q = np.eye(d)
    
    """
    H = I - 1/n * 1 * 1.T
    """
    H = np.eye(n,n) - 1/n*np.ones((n,n))
    
    
    gama = 10^-6
    """
    W = (X*H*X.T + gama * Q)^-1 * X*H*Y
    """
    W = np.dot( np.dot( np.dot( \
                np.linalg.inv( np.dot( np.dot(X,H), X.T)+2*Q ) \
                               , X), H), Y)
    """
    q(ij) = ||W||2,1 / ||w^j||2
    
    axis = 1 =》 对W的每一行求L2范数
    np.linalg.norm(W, ord=2, axis = 1) =》 对每行求L2范数
    """
    Q = np.sum(np.linalg.norm(W, ord=2, axis = 1)) / np.linalg.norm(W, ord = 2, axis=1)
    Q = np.diag(Q)
    
    
    
    
    pre_f = cur_f = fun22_value(W, X, H, Q, Y)    
    
    NITER = 900
    epsilon = 10^-6
    for i in range(NITER):
        pre_f = cur_f
        W = np.dot( np.dot( np.dot( \
                np.linalg.inv( np.dot( np.dot(X,H), X.T)+2*Q ) \
                               , X), H), Y)
        Q = np.diag(np.linalg.norm(W, axis=1))
        cur_f = fun22_value(W, X, H, Q, Y)
        if (cur_f - pre_f) / cur_f < epsilon:
            break
    return W



file_path = "..\\..\\data_selected\\gene\\brain\\"

selected_data_file_name = "selected_data"
# selected_feature_file_name = "selected_features"
selected_cluster_name_file_name = "selected_cluster_names"

unselected_data_file_name = "unselected_data"
# unselected_feature_file_name = "unselected_features"
# unselected_cluster_name_file_name = "unselected_cluster_names"

example_rate = 50
feature_rate = 1
unselected_example_rate = 100 - example_rate
unselected_feature_rate = 100 - feature_rate
selected_data_file_name = file_path + selected_data_file_name + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"
# selected_feature_file_name = file_path + selected_feature_file_name + "_" + str(feature_rate) + "" + ".txt"
selected_cluster_name_file_name = file_path + selected_cluster_name_file_name + "_"  + str(example_rate) + "" + ".txt"

unselected_data_file_name = file_path + unselected_data_file_name + "_" +  str(unselected_example_rate) + "_" + str(feature_rate) + "" + ".txt"

output_file_name = file_path + "result" + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"

print(selected_data_file_name)
# print(selected_feature_file_name)
print(selected_cluster_name_file_name)
print(unselected_data_file_name)
print(output_file_name)


# 读入数据，分别是带标签，不带标签和标签
# 读入数据，分别是带标签，不带标签和标签
XL = pd.read_csv(selected_data_file_name,sep=",",header=None, engine='python')
YL = pd.read_csv(selected_cluster_name_file_name,sep=",",header=None, engine='python')
XU = pd.read_csv(unselected_data_file_name,sep=",",header=None, engine='python')


# 类型转换
XL = XL.values
YL = YL.values
XU = XU.values

# 要改成一维的，否则会算法会报错，因为取出的每个特征也是一维的
YL = np.reshape(YL, (1,-1))[0]



# 要改成一维的，否则会算法会报错，因为取出的每个特征也是一维的
YL = np.reshape(YL, (1,-1))[0]

c = np.max(YL) + 1
ln, d = XL.shape
un =  XL.shape[0]

tmp_YL = np.zeros((ln, c))
for i,j in enumerate(YL):
    tmp_YL[i,j] = 1
YL = tmp_YL



start_time = time.clock()

X = XL.T # X : (d X n)
Y = YL
n = X.shape[1]
W = get_W(X, Y)
b = 1/n*(np.dot(Y.T, np.ones((n,1))) - np.dot(np.dot(W.T, X), np.ones((n,1))))

X = XU.T
YU = np.zeros((un, c))
for i in range(X.T.shape[1]):
    ad = np.dot(X[:,i:i+1].T, W) + b.T
    ad_new, ft = EProjSimplex_new(ad)
    YU[i:i+1,:] = ad_new.A


X = sp.concatenate((XL, XU), axis = 0)
Y = sp.concatenate((YL, YU), axis = 0)


X = X.T
cur_f = fun8_value(X, Y, W, b)


NITER = 10
epsilon = 10^-6
for i in range(NITER):
    pre_f = cur_f

    W = get_W(X, Y)
    n = X.shape[1]
    
#     print(Y.T.shape, np.ones((n,1)).shape , W.T.shape, X.T.shape, np.ones((n,1)).shape)
    b = 1/n*(np.dot(Y.T, np.ones((n,1))) - np.dot(np.dot(W.T, X), np.ones((n,1))))

    X = XU.T
    for i in range(X.T.shape[1]):
        """
        min ( ||(xi.T) * W + b.T - yi.T||的F范数 ^ 2 )
        s.t. yi>=0, 1*yi=1
        """
        ad = np.dot(X[:,i:i+1].T, W) + b.T
        ad_new, ft = EProjSimplex_new(ad)
        YU[i:i+1,:] = ad_new.A


    X = sp.concatenate((XL, XU), axis = 0)
    Y = sp.concatenate((YL, YU), axis = 0)
    
    X = X.T
    cur_f = fun8_value(X, Y, W, b)

    if (cur_f - pre_f) / cur_f < epsilon:
        break
        
        
s = np.zeros(d)
W_L2_sum = np.sum([np.linalg.norm(w, ord=2) for w in W])
s = np.linalg.norm(W, ord=2, axis = 1) / W_L2_sum
feature_order = list( np.argsort(s) )

print(time.clock() - start_time)
    
    
s = np.zeros(d)
W_L2_sum = np.sum([np.linalg.norm(w, ord=2) for w in W])
s = np.linalg.norm(W, ord=2, axis = 1) / W_L2_sum
feature_order = list( np.argsort(s) )


with open(output_file_name, "w+") as result_file:
    print("\n".join([str(w) for w in feature_order]), file=result_file)
print(feature_order)

import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time
import sys
import matplotlib.pyplot as plt


from LSFS_FUN import *
from LSFS_TEST import *
from read_data import *
from evaluate import *
from my_pickle import to_pickle


def append_module_path():
    import sys
    paths = [ \
        "../gen_data",
        "../evaluate",
        "../read_data",
        "../PRPC",
        "../LSDF",
        "../laplacian_score",
        "../Fisher",
        "../sSelect"
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
import sSelect_FUN
import fisher_score_FUN


def plot_theta_feature(theta_feature_table, xlabel = "", ylabel = "$\\theta$", feature_num = 10, theta_feature_picture_path=None):
    """
    行：算法名
    列：特征个数
    值：accuracy
    """
    style = ['.--','o--','v--','s--','p--','*--','h--', 'H--','+--','x--','D--']
    
#     style = ['--','--','--','--','--','--','--', '--','--','--','--']

    labels = theta_feature_table.index.get_values()
#     x = theta_feature_table.columns.get_values()
    x = list( range(feature_num) )
    
    plt.figure(figsize=(10, 8))
    plt.xlim(min(x), max(x) + 1 )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for i,label in enumerate(labels):
        row_data = theta_feature_table.ix[label, :].get_values()
#         print(label, row_data, x, style[i])
        
        plt.plot(x, sorted(list(row_data))[::-1][:feature_num], style[i], label = label)

    plt.legend(loc='best')
    if theta_feature_picture_path != None:
        plt.savefig(theta_feature_picture_path)

        
def cal_theta_feature_tabel(XL, YL, XU, gama_array, theta_feature_table_path):
    
    theta_feature_list = []
    
    n, m = XL.shape
    
    for gama in gama_array:
        print(str(gama) + " start")
        theta_feature = lsfs_theta(XL, YL, XU, gama = gama)
        theta_feature_list.append(theta_feature)
        
    theta_feature_table = pd.DataFrame(data=theta_feature_list, index=gama_array, columns=list(range(m)))
    
    if theta_feature_table_path != None:
        theta_feature_table.to_csv(theta_feature_table_path)
        
    return theta_feature_table



def lsfs_theta(XL, YL, XU, gama = 10**-6):
    
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
    
    return s



file_path = "..\\..\\data_selected\gene\\prostate\\"
print(file_path.split("\\")[-2] + " start")

gama = 0.1
step = 5

selected_data_file_name = "selected_data"
# selected_feature_file_name = "selected_features"
selected_cluster_name_file_name = "selected_cluster_names"

unselected_data_file_name = "unselected_data"
# unselected_feature_file_name = "unselected_features"
unselected_cluster_name_file_name = "unselected_cluster_names"
example_rate = 30
feature_rate = 10


theta_feature_table_path = "..\\result\\thea_feature_table\\" + "acc_" + file_path.split("\\")[-2] + "_" +  \
                                "gama_" + str(gama) + "_" + \
                            str(example_rate) + "_" + str(feature_rate) + ".csv"



theta_feature_picture_path = "..\\result\\thea_feature\\" + "acc_" + file_path.split("\\")[-2] + "_" +  \
                            "gama_" + str(gama) + "_" + \
                            str(example_rate) + "_" + str(feature_rate) + ".png"



# idx_array = np.array([step*i for i in range(1, 8)])

# xlabel = "Features id"
xlabel = ""
ylabel = "$\\theta$"


gama_array = [10**i for i in range(-3, 4)]


XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                    unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)


XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()

YL = read_data.label_n1_to_nc(YL)
YU = read_data.label_n1_to_nc(YU)

# feature_num = XU.shape[1]
feature_num = 20

theta_feature_table = cal_theta_feature_tabel(XL, YL, XU, gama_array, \
                                              theta_feature_table_path=theta_feature_table_path)

plot_theta_feature(theta_feature_table, xlabel, ylabel, feature_num = feature_num, theta_feature_picture_path = theta_feature_picture_path)
print(file_path.split("\\")[-2] + " finished")
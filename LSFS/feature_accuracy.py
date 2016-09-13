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



def plot_acc_arr(acc_array_table, xlabel = "Numbers of Selected Features", ylabel = "Accuracy", acc_array_picture_path=None):
    """
    行：算法名
    列：特征个数
    值：accuracy
    """
    style = ['.--','o--','v--','s--','p--','*--','h--', 'H--','+--','x--','D--']

    labels = acc_array_table.index.get_values()
    x = acc_array_table.columns.get_values()
    
    plt.figure(figsize=(10, 8))
    plt.xlim(min( acc_array_table.columns.get_values()), max( acc_array_table.columns.get_values()) + 1 )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for i,label in enumerate(labels):
        row_data = acc_array_table.ix[label, :].get_values()
#         print(label, row_data, x, style[i])
        plt.plot(x, row_data, style[i], label = label)

    plt.legend(loc='best')
    if acc_array_picture_path != None:
        plt.savefig(acc_array_picture_path)

        
        
        
        
def cal_acc_tabel(feature_order_table, idx_array, acc_array_table_path):
    acc_array_list = []
    
    labels = feature_order_table.index.get_values()
    
    for label in labels:
        feature_order = feature_order_table.ix[label, :]
        """cal_many_acc_by_idx2不抽样，cal_many_acc_by_idx抽样"""
        acc_array = evaluate.cal_many_acc_by_idx2(XL_train, YL_train, XU_train, YU_train,\
                                   feature_order.get_values(), idx_array = idx_array)
        acc_array_list.append(acc_array)
#         print(acc_array)
        
    acc_array_table = pd.DataFrame(data=acc_array_list, index=labels, columns=idx_array)
    
    if acc_array_table_path != None:
        acc_array_table.to_csv(acc_array_table_path)
    return acc_array_table


def cal_feature_order_table(XL_train, YL_train, XU_train, YU_train,fun_list, gama_list, feature_order_table_path = None):
    exc_fun_label = []
    fn = XL_train.shape[1]
    xf = range(fn)
    feature_order_list = []
    for fun in fun_list:
        XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
        fun_name = fun.__name__
        if fun_name == "lsfs":
            print("lsfs feature select start")
            YL = read_data.label_n1_to_nc(YL)
            YU = read_data.label_n1_to_nc(YU)
            for gama in gama_list:
                feature_order, time_dual = lsfs(XL, YL, XU, gama = gama)
                exc_fun_label.append("LSFS :" + str(gama))
                feature_order_list.append(feature_order)
                with open("time.txt", "a+") as f:
                    print(fun_name + " " + str(gama) + " time : " +  str(time_dual), file=f)

            
        elif fun_name == "fisher_score":
            print("fisher_score feature select start")
        #     feature_order, time_dual = fisher_score(XL_train, YL_train, output_file_name=output_file_name)
            feature_order, time_dual = fun(XL, YL)
            exc_fun_label.append("Fisher score")
            with open("time.txt", "a+") as f:
                print(fun_name + " time : " +  str(time_dual), file=f)


        elif fun_name == "laplacian_score_feature_order2":
            print("laplacian_score_feature_order2 feature select start")
        #     feature_order, time_dual = laplacian_score_feature_order2(XL, k = 10, output_file_name=output_file_name)
            feature_order, time_dual = fun(XL, k = 10)
            exc_fun_label.append("Laplacian score")
            with open("time.txt", "a+") as f:
                print(fun_name + " time : " +  str(time_dual), file=f)


        elif fun_name == "lsdf":
            print("lsdf feature select start")
        #     feature_order, time_dual =  lsdf(XL, YL, XU, output_file_name=output_file_name)
            feature_order, time_dual = fun(XL, YL, XU)
            exc_fun_label.append("LSDF")
            with open("time.txt", "a+") as f:
                print(fun_name + " time : " +  str(time_dual), file=f)


        elif fun_name == "prpc":
            print("prpc feature select start")
            feature_order, time_dual = fun(XL, YL, XU)
            exc_fun_label.append("PRPC")
            with open("time.txt", "a+") as f:
                print(fun_name + " time : " +  str(time_dual), file=f)


        elif fun_name == "sSelect":
            print("sSelect feature select start")
            feature_order, time_dual = fun(XL, YL, XU, k = 10, theta = 10, namuda = 0.1)
            exc_fun_label.append("sSelect")
            with open("time.txt", "a+") as f:
                print(fun_name + " time : " +  str(time_dual), file=f)


        if fun_name != "lsfs":
            feature_order_list.append(feature_order)
    #     print(feature_order)
    
    with open("time.txt", "a+") as f:
        print("======================================================================", file=f)
    
    print("finished feature select!")
#     print(exc_fun_label)

    feature_order_table = pd.DataFrame(data=np.array(feature_order_list), index=exc_fun_label, columns=xf)
    if feature_order_table_path != None:
        feature_order_table.to_csv(feature_order_table_path)
    return feature_order_table



file_paths = [
#     "..\\..\\data_selected\gene\\adenocarcinoma\\",
#    "..\\..\\data_selected\\gene\\brain\\",
    "..\\..\\data_selected\\gene\\breast.2.class\\",
    "..\\..\\data_selected\\gene\\breast.3.class\\",
#    "..\\..\\data_selected\\gene\\colon\\",
#    "..\\..\\data_selected\\gene\\leukemia\\",
#    "..\\..\\data_selected\\gene\\lymphoma\\",
#    "..\\..\\data_selected\\gene\\nci\\",
#    "..\\..\\data_selected\\gene\\prostate\\",
#    "..\\..\\data_selected\\gene\\srbct\\"
#     "..\\..\\data_selected\\gene\\MADELON\\"
]

for file_path in file_paths:
    gama_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    step = 10
    
    selected_data_file_name = "selected_data"
    # selected_feature_file_name = "selected_features"
    selected_cluster_name_file_name = "selected_cluster_names"
    
    unselected_data_file_name = "unselected_data"
    # unselected_feature_file_name = "unselected_features"
    unselected_cluster_name_file_name = "unselected_cluster_names"
    example_rate = 30
    feature_rate = 100
    
    
    feature_order_table_path = "..\\result\\feature_order_table2\\" + "prpc_" + file_path.split("\\")[-2] + "_" + \
                                str(example_rate) + "_" + str(feature_rate) + ".csv"
    

                                

    acc_array_table_path = "..\\result\\acc_array_table2\\" + "prpc_acc_" + file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate) + ".csv"
    

                                
        
    acc_array_picture_path = "..\\result\\acc_array_picture2\\" + "prpc_acc_" + file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate) + ".png"

                                
        
    xlabel = "Numbers of Selected Features"
    ylabel = "Accuracy"
        
    fun_list = []
    fun_list.append(lsfs)
#    fun_list.append(PRPC_FUN.prpc)
    fun_list.append(laplacian_score_FUN.laplacian_score_feature_order2)
    fun_list.append(LSDF_FUN.lsdf)
    fun_list.append(sSelect_FUN.sSelect)
    fun_list.append(fisher_score_FUN.fisher_score)

    
    XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                        unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)
    
    
    feature_order_table = cal_feature_order_table(XL_train, YL_train, XU_train, YU_train, \
                                              fun_list, gama_list=gama_list, \
                                              feature_order_table_path = feature_order_table_path)
    
    idx_array = np.array([step*i for i in range(1, 20)])
    
    acc_array_table = cal_acc_tabel(feature_order_table, idx_array, acc_array_table_path)
#    plot_acc_arr(acc_array_table, xlabel, ylabel, acc_array_picture_path = acc_array_picture_path)

print("finished")



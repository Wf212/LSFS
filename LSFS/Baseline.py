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
    
    
file_paths = [
#     "..\\..\\data_selected\gene\\adenocarcinoma\\",
#    "..\\..\\data_selected\\gene\\brain\\",
#    "..\\..\\data_selected\\gene\\breast.2.class\\",
    "..\\..\\data_selected\\gene\\breast.3.class\\",
    "..\\..\\data_selected\\gene\\colon\\",
#    "..\\..\\data_selected\\gene\\leukemia\\",
    "..\\..\\data_selected\\gene\\lymphoma\\",
#    "..\\..\\data_selected\\gene\\nci\\",
    "..\\..\\data_selected\\gene\\prostate\\",
#    "..\\..\\data_selected\\gene\\srbct\\",
#     "..\\..\\data_selected\\gene\\MADELON\\"
]

for file_path in file_paths:
    gama_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    selected_data_file_name = "selected_data"
    # selected_feature_file_name = "selected_features"
    selected_cluster_name_file_name = "selected_cluster_names"
    
    unselected_data_file_name = "unselected_data"
    # unselected_feature_file_name = "unselected_features"
    unselected_cluster_name_file_name = "unselected_cluster_names"
    example_rate = 40
    feature_rate = 10
    

    acc_array_table_path = "..\\result\\acc_array_table2\\" + "baseline_acc_" + file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate) + ".csv"
    

    acc_array_picture_path = "..\\result\\acc_array_picture2\\" + "baseline_acc_" + file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate) + ".png"


    XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                        unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)
    
    
    idx_array = np.array([XL_train.shape[1]])
    feature_order_table = pd.DataFrame(data=np.array([list(range(XL_train.shape[1]))]), index=["Baseline"], columns=list(range(XL_train.shape[1])))
    
    acc_array_table = cal_acc_tabel(feature_order_table, idx_array, acc_array_table_path)
    
    print(file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate) + "   ", acc_array_table.get_values())
    

print("finished")




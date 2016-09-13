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




def cal_all_gama_acc_table(XL_train, YL_train, XU_train, YU_train, feature_order_table, acc_array_table_path):
    acc_array_list = []
    
    labels = feature_order_table.index.get_values()
    
    idx_array =  np.array(list(range(1, feature_order_table.columns.size)))
#     idx_array =  np.array([1,3,5])
    
    for label in labels:
        feature_order = feature_order_table.ix[label, :]
#         print(feature_order.get_values())
#         print(idx_array)
        acc_array = evaluate.cal_many_acc_by_idx2(XL_train, YL_train, XU_train, YU_train,\
                                   feature_order.get_values(), idx_array = idx_array)
        acc_array_list.append(acc_array)
        
    acc_array_table = pd.DataFrame(data=acc_array_list, index=labels, columns=idx_array)
    
    if acc_array_table_path != None:
        acc_array_table.to_csv(acc_array_table_path)
    return acc_array_table



# fun = lsfs
# fun = PRPC_FUN.prpc
# fun = laplacian_score_FUN.laplacian_score_feature_order2
# fun = LSDF_FUN.lsdf
# fun = sSelect_FUN.sSelect
# fun = fisher_score_FUN.fisher_score

def cal_gama_feature_table(XL_train, YL_train, XU_train, YU_train, gama_list, gama_feature_table_path = None):
    gama_label = []
    fn = XL_train.shape[1]
    xf = range(fn)
    feature_order_list = []
    for gama in gama_list:
        XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
        print( str(gama) + " : lsfs feature select start")
        YL = read_data.label_n1_to_nc(YL)
        YU = read_data.label_n1_to_nc(YU)
        feature_order, time_dual = lsfs(XL, YL, XU, gama = gama)
        feature_order_list.append(feature_order)
        gama_label.append(str(gama))

    print("finished feature select!")
#     print(exc_fun_label)

    gama_feature_table = pd.DataFrame(data=np.array(feature_order_list), index=gama_label, columns=xf)
    if gama_feature_table_path != None:
        gama_feature_table.to_csv(gama_feature_table_path)
    return gama_feature_table


file_paths = [
#     "..\\..\\data_selected\gene\\adenocarcinoma\\",
#    "..\\..\\data_selected\\gene\\brain\\",
#    "..\\..\\data_selected\\gene\\breast.2.class\\",
#    "..\\..\\data_selected\\gene\\breast.3.class\\",
#    "..\\..\\data_selected\\gene\\colon\\",
#    "..\\..\\data_selected\\gene\\leukemia\\",
#    "..\\..\\data_selected\\gene\\lymphoma\\",
#    "..\\..\\data_selected\\gene\\nci\\",
    "..\\..\\data_selected\\gene\\prostate\\",
#    "..\\..\\data_selected\\gene\\srbct\\"
]

for file_path in file_paths:
    selected_data_file_name = "selected_data"
    # selected_feature_file_name = "selected_features"
    selected_cluster_name_file_name = "selected_cluster_names"
    
    unselected_data_file_name = "unselected_data"
    # unselected_feature_file_name = "unselected_features"
    unselected_cluster_name_file_name = "unselected_cluster_names"
    example_rate = 30
    feature_rate = 10
    
    print(file_path.split("\\")[-2]  +  "  start")
    
    XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                        unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)
    
    # feature_order, time_dual, a = run_accuracy(lsfs, XL_train,YL_train,XU_train,YU_train, 10, output_file_name)

    gama_feature_table_path = "..\\result\\feature_order_table\\" + "all_" + file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate) + ".csv"

    gama_acc_table_path = "..\\result\\acc_array_table\\" + "all_acc_" + file_path.split("\\")[-2] + "_" +  \
                                        str(example_rate) + "_" + str(feature_rate) + ".csv"
        
    # acc_array_picture_path = "..\\result\\acc_array_picture\\" + "acc_" + file_path.split("\\")[-2] + "_" +  \
    #                                 str(example_rate) + "_" + str(feature_rate) + ".png"
        
    xlabel = "Numbers of Selected Features"
    ylabel = "Accuracy"
        
        
    gama_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    feature_order_table = cal_gama_feature_table(XL_train, YL_train, XU_train, YU_train, \
                                              gama_list, \
                                              gama_feature_table_path = gama_feature_table_path)
    
    gama_acc_table = cal_all_gama_acc_table(XL_train, YL_train, XU_train, YU_train, feature_order_table, gama_acc_table_path)
    
print("finished")





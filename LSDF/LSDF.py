#!usr/bin/python
# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
import scipy as sp
import sys
import time

from LSDF_FUN import *

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
import gen_data
import evaluate
import read_data
from my_pickle import to_pickle

file_path = "..\\..\\data_selected\\gene\\brain\\"

selected_data_file_name = "selected_data"
# selected_feature_file_name = "selected_features"
selected_cluster_name_file_name = "selected_cluster_names"

unselected_data_file_name = "unselected_data"
# unselected_feature_file_name = "unselected_features"
unselected_cluster_name_file_name = "unselected_cluster_names"
example_rate = 50
feature_rate = 10

output_file_name = "..\\result\\" + "lsdf_result" + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"

XL_train, YL_train, XU_train, YU_train  = read_data.get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                    unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)

# feature_order, time_dual, a = run_accuracy(lsfs, XL_train,YL_train,XU_train,YU_train, 10, output_file_name)

XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()

feature_order, time_dual =  lsdf(XL, YL, XU, output_file_name=output_file_name)

num_feature = len(feature_order)
#if num_feature > 300:
#    num_feature = 300

#feature_order = np.sort(feature_order)

acc_array = evaluate.cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = num_feature)


XL_train2, YL_train2, XU_train2, YU_train2  = read_data.get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                    unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)

to_pickle("..\\result\\"+"lsdf_accuracy" + "_" +  str(example_rate) + "_" + str(feature_rate) + ".pkl", acc_array)

print(feature_order)
print("===================================================================")
print("===================================================================")
# print("accuracy : ", a)
print("time : ", time_dual)


evaluate.plot_array_like(acc_array, xlabel_name="number feature", ylabel_name="accuracy")


print(np.sum(XL_train2 != XL_train), np.sum(YL_train2 != YL_train), np.sum(XU_train2 != XU_train), np.sum(YU_train2 != YU_train))

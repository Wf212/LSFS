#!/usr/bin/python
# -*- coding:utf-8 -*- 

import numpy as np
import pandas as pd

def read_feature(path, sep=",", header=None, engine="python"):
    data = pd.read_csv(path,sep=sep,header=header, engine=engine)
    data = data.values
    return data

def read_label(path, sep=",", header=None, engine="python"):
    """
    读入二维 n x 1 数据
    转换为一维 n 数据
    """
    data = pd.read_csv(path,sep=",",header=None, engine='python')
    data = data.values
    data = np.reshape(data, (1,-1))[0]
    return data



def label_n1_to_nc(label):
    """
    一维 n => 二维 n x c
    """
    c = int(np.max(label)) + 1
    n = len(label)
    tmp_label = np.zeros((n, c))
    for i,j in enumerate(label):
        tmp_label[int(i),int(j)] = 1
    label = tmp_label
    return label



def label_nc_to_n1(label):
    """
    一维 n => 二维 n x c
    """
    n = len(label)
    tmp_label = np.zeros(n)
    for i,arr in enumerate(label):
        for j,v in enumerate(arr):
            if v > 0:
                tmp_label[i] = j
    label = tmp_label
    return label



def get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
             unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate):

    # unselected_example_rate = 100 - example_rate
    # unselected_feature_rate = 100 - feature_rate
    selected_data_file_name = file_path + selected_data_file_name + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"
    # selected_feature_file_name = file_path + selected_feature_file_name + "_" + str(feature_rate) + "" + ".txt"
    selected_cluster_name_file_name = file_path + selected_cluster_name_file_name + "_"  + str(example_rate) + "_" + str(feature_rate) + ".txt"

    unselected_data_file_name = file_path + unselected_data_file_name + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"
    unselected_cluster_name_file_name = file_path + unselected_cluster_name_file_name + "_"  + str(example_rate) + "_" + str(feature_rate) + ".txt"
    

    print(selected_data_file_name)
    # print(selected_feature_file_name)
    print(selected_cluster_name_file_name)
    print(unselected_data_file_name)
    print(unselected_cluster_name_file_name)


    # 读入数据，分别是带标签，不带标签和标签
    # 读入数据，分别是带标签，不带标签和标签
    XL = pd.read_csv(selected_data_file_name,sep=",",header=None, engine='python')
    YL = pd.read_csv(selected_cluster_name_file_name,sep=",",header=None, engine='python')
    XU = pd.read_csv(unselected_data_file_name,sep=",",header=None, engine='python')
    YU = pd.read_csv(unselected_cluster_name_file_name,sep=",",header=None, engine='python')


    # 类型转换
    XL = XL.values
    YL = YL.values
    XU = XU.values
    YU = YU.values

    # 要改成一维的，否则会算法会报错，因为取出的每个特征也是一维的
    YL = np.reshape(YL, (1,-1))[0]
    YU = np.reshape(YU, (1,-1))[0]
    return XL, YL, XU, YU

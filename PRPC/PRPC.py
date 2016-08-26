#!/usr/bin/python
# -*- coding:utf-8 -*- 

import pandas as pd
import scipy as sp
import numpy as np
import time
from my_math import *

"""
输出特征的索引编号，输出顺序是被选择的顺序
"""

file_path = "..\\..\\data_selected\\gene\\brain\\"

selected_data_file_name = "selected_data"
# selected_feature_file_name = "selected_features"
selected_cluster_name_file_name = "selected_cluster_names"

unselected_data_file_name = "unselected_data"
# unselected_feature_file_name = "unselected_features"
# unselected_cluster_name_file_name = "unselected_cluster_names"

example_rate = 50
feature_rate = 5
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
label_data = pd.read_csv(selected_data_file_name,sep=",",header=None, engine='python')
label = pd.read_csv(selected_cluster_name_file_name,sep=",",header=None, engine='python')
unlabel_data = pd.read_csv(unselected_data_file_name,sep=",",header=None, engine='python')


# 类型转换
label_data = label_data.values
label = label.values
unlabel_data = unlabel_data.values

# 要改成一维的，否则会算法会报错，因为取出的每个特征也是一维的
YL = np.reshape(label, (1,-1))[0]

F = set(range(label_data.shape[1]))
Fs = set()
order_Fs = []
Fa = F - Fs

# 选取的特征个数
S = len(F)

len_label_data = label_data.shape[0]
data = sp.concatenate((label_data, unlabel_data), axis = 0)

s = time.clock()

for k in range(1,S+1):
    max_pearson = float("-inf")
    Fk = -1
    
    
    
    for Fj in Fa:
        
        
        
        pearson1 = compute_pearson(data[:len_label_data,Fj], YL)
        

#         print(label_data[:,Fj])
#         print(YL)
#         print(pearson1)
        
#         print(Fj)

        
        pearson2_sum = 0
        
        for Fi in Fs:
            pearson2 = compute_pearson(data[:,Fj], data[:,Fi])
            pearson2_sum += pearson2
        
        
        if k == 1:
            pearson2_sum = 0
        elif k > 1:
            pearson2_sum /= k - 1
            
#         print(pearson1, pearson2_sum)
        
        pearson1 -= pearson2_sum
        
        if pearson1 > max_pearson:
            max_pearson = pearson1
            Fk = Fj
    
    
    
    Fs = Fs | {Fk}
    Fa = Fa - {Fk}
    
    order_Fs.append(Fk)

print("time: ", time.clock() - s)

with open(output_file_name, "w+") as result_file:
    print("\n".join([str(w) for w in order_Fs]), file=result_file)
print(order_Fs)
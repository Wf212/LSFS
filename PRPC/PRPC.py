import pandas as pd
import scipy as sp
import numpy as np
from my_math import *

"""
输出特征的索引编号，输出顺序是被选择的顺序
"""

# 读入数据，分别是带标签，不带标签和标签
label_data = pd.read_csv("labeled_data.txt",sep="\s*",header=None, engine='python')
label = pd.read_csv("label.txt",sep="\s*",header=None, engine='python')
unlabel_data = pd.read_csv("unlabeled_data.txt",sep="\s*",header=None, engine='python')

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
S = len(F)

for k in range(1,S+1):
    for Fj in Fa:
        max_pearson = float("-inf")
        Fk = -1
        pearson1 = compute_pearson(label_data[:,Fj], YL)
#         print(label_data[:,Fj])
#         print(YL)
#         print(pearson1)
        
#         print(Fj)
        Fj_data = sp.concatenate((label_data[:,Fj], unlabel_data[:,Fj]), axis = 0)
        pearson2_sum = 0
        
        for Fi in Fs:
            Fi_data = sp.concatenate((label_data[:,Fi], unlabel_data[:,Fi]), axis = 0)
            pearson2 = compute_pearson(Fj_data, Fi_data)
            pearson2_sum += pearson2
        
        if k == 1:
            pearson2_sum = 0
        elif k > 1:
            pearson2_sum /= k - 1
            
        pearson1 += pearson2_sum
        
        if pearson1 > max_pearson:
            pearson1 = max_pearson
            Fk = Fj
    
    Fs = Fs | {Fk}
    Fa = Fa - {Fk}
    order_Fs.append(Fk)
print(order_Fs)
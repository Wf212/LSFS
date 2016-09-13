#/usr/bin/python
# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
import sys


def partition(arr, low, high):
    key = arr[low]
    while (low < high):
        while (low < high and arr[high] >= key): high -= 1
        arr[low] = arr[high]
        while (low < high and arr[low] <= key): low += 1
        arr[high] = arr[low] 
    arr[low] = key
    return low


def quicksort(arr, left, right):
    # 只有left < right 排序
    if left < right:
        pivot_index = partition(arr, left, right)
        quicksort(arr, left, pivot_index - 1)
        quicksort(arr, pivot_index + 1, right)

        

def filter_KNN(S, k = 1):
    """
    S矩阵的每行留下最小的K个值
    """
    
    """
    # 注意：小s是大S的每行的引用
    """
    S = S.copy()
    for i,s in enumerate(S):
        s = s.copy()
        k_value = k_th(s, 0, len(s)-1, k)
        
        need_cnt = k - ( len(s) - np.sum(s>=k_value) )
    
        S[i:i+1,s>k_value] = 0
        
        index = []
        for j in range(len(s)):
            if s[j] == k_value:
                index.append(j)
    
        if (len(index) < need_cnt):
            print("equal number invalid")
            
#         print(index)
        np.random.shuffle(index)
        S[i:i+1, index[:len(index) - need_cnt]] = 0
    
    return S


def get_KNN_flag(S, k = 1):
    """
    S矩阵的每行留下最小的K个值
    """
    
    """
    # 注意：小s是大S的每行的引用
    如果k大于样本数，则k会被设置成样本数的一半
    """
    
    S = S.copy()
    n, m = S.shape
    knn_flag = np.ones((n, m))
    
    if m < k:
        print("k大于样本数，k被设置成样本数的一半")
        k = int( m * 0.5 )
    
#    for i in range(np.min([n, m])):
#        S[i, i] = np.inf
    
    for i,s in enumerate(S):
        s = s.copy()
        k_value = k_th(s, 0, len(s)-1, k)
        
        need_cnt = k - ( len(s) - np.sum(s>=k_value) )
    
        knn_flag[i:i+1,s>k_value] = 0
        
        index = []
        for j in range(len(s)):
            if s[j] == k_value:
                index.append(j)
    
        if (len(index) < need_cnt):
            print("equal number invalid")
            
#         print(index)
        np.random.shuffle(index)
        knn_flag[i:i+1, index[:len(index) - need_cnt]] = 0
    
    return knn_flag.astype(np.bool)


def k_th(arr, low, high, k, it_cnt = None):
    import copy
    """
    寻找前K小
    """
    # 这步不能少，否则会改变外部数据
    if it_cnt == None:
        it_cnt = 1
        arr = copy.deepcopy(arr)
    
    it_cnt += 1
    
    if (high - low + 1 < k or k < 1): 
        print("error : high - low + 1 < k or k < 1")
        return -1
    pos = partition(arr, low, high)
    cnt = pos - low + 1
    if (cnt == k): 
        return arr[pos]
    elif (cnt < k):
        return k_th(arr, pos + 1, high, k - cnt, it_cnt)
    else:
        return k_th(arr, low, pos, k, it_cnt)


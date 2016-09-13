#!/usr/bin/python
# -*- coding:utf-8 -*- 

import pandas as pd
import scipy as sp
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D   

def accuracy(true_label, clust_label):
    """
    true_label : 想要的结果
    clust_label : 算法的预测
    """
    tp = tn = 0
    fp = fn = 0
    for i in range( len(true_label) ):
        for j in range( len(clust_label) ):
            # true positive
            if (true_label[i] == true_label[j] and clust_label[i] == clust_label[j]):
                tp+=1
            # true negative 
            elif (true_label[i] != true_label[j] and clust_label[i] != clust_label[j]):
                tn+=1
            # false positive
            elif (true_label[i] != true_label[j] and clust_label[i] == clust_label[j]):
                fp+=1
            else:
            # false negative
                fn+=1
    return (tp+tn)*1.0/(tp+tn+fp+fn)


def select_data(XL, YL, XU, YU, feature_order, sel_num = None):
    """
    重组和筛选特征
    """
    if sel_num == None:
        sel_num = len(feature_order)
        
    X = sp.concatenate((XL, XU), axis = 0)
    sel_feature = feature_order[:sel_num]
    X = X[:,sel_feature]
    Y = sp.concatenate((YL, YU), axis = 0)
    return X, Y


def select_data2(XL, YL, XU, YU, feature_order, sel_num = None):
    """
    重组和筛选特征
    """
    if sel_num == None:
        sel_num = len(feature_order)
    sel_feature = feature_order[:sel_num]
    return XL[:,sel_feature].copy(), YL.copy(), XU[:,sel_feature].copy(), YU.copy()


def get_train_test_rate(x, y, rate=None):
    """
    抽样=》训练，测试
    x 特征 （样本个数 x 特征个数）
    y 类别
    """
    n_samples = len(x)
    if rate == None:
        train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
        x_train, y_train = x[train_mask, :], y[train_mask]
        x_test, y_test = x[~train_mask, :], y[~train_mask]
    else:
        row_selected_rate = rate
        row_selected = random.sample(range(n_samples), int(row_selected_rate*n_samples))
        row_unselected = list(set(range(n_samples)) - set(row_selected))
        
        x_train, y_train = x[row_selected, :], y[row_selected]
        x_test, y_test = x[row_unselected, :], y[row_unselected]
        
    return x_train, y_train, x_test, y_test


def run_acc(x, y, row_selected_rate=None):
    cnt = 10
    sum_accuracy = 0.0
    avg_accuracy = 0.0
    model = svm.LinearSVC(loss='hinge')
    for i in range(cnt):
        row_num = x.shape[0]
        # 根据比例筛选数据
        row_selected_rate = 0.5
#         row_selected = random.sample(range(row_num), int(row_selected_rate*row_num))
#         row_unselected = list(set(range(row_num)) - set(row_selected))
        
#         train_set = X[row_selected, :]
#         train_label = Y[row_selected]
#         test_set = X[row_unselected, :]
#         test_label = Y[row_unselected]
        
        train_set, train_label, test_set, test_label = get_train_test_rate(x, y, rate=row_selected_rate)
        
        
        # fit a SVM model to the data
        model.fit(train_set, train_label)
        # print(model)
        # make predictions
        expected = test_label
        predicted = model.predict(test_set)
        
        sum_accuracy += accuracy(expected, predicted)
    avg_accuracy = sum_accuracy / cnt
    return avg_accuracy


def run_acc2(train_set, train_label, test_set, test_label):
    cnt = 10
    sum_accuracy = 0.0
    avg_accuracy = 0.0
    model = svm.LinearSVC(loss='hinge')
    for i in range(cnt):
#        row_num = x.shape[0]
        # 根据比例筛选数据
#        row_selected_rate = 0.5
#         row_selected = random.sample(range(row_num), int(row_selected_rate*row_num))
#         row_unselected = list(set(range(row_num)) - set(row_selected))
        
#         train_set = X[row_selected, :]
#         train_label = Y[row_selected]
#         test_set = X[row_unselected, :]
#         test_label = Y[row_unselected]
        
        # fit a SVM model to the data
        model.fit(train_set, train_label)
        # print(model)
        # make predictions
        expected = test_label
        predicted = model.predict(test_set)
        
        sum_accuracy += accuracy(expected, predicted)
    avg_accuracy = sum_accuracy / cnt
    return avg_accuracy



def cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = 10):
    acc_array = np.zeros(num_feature)
    for i in range(1,num_feature):
        X,Y = select_data(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, sel_num=i)
        a = run_acc(X,Y)
        acc_array[i] = a
#        print(a)
    return acc_array


def cal_many_acc2(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = 10):
    acc_array = np.zeros(num_feature)
    for i in range(1,num_feature):
#        X,Y = select_data(XL_train, YL_train, XU_train, YU_train,\
#                           feature_order, sel_num=i)
#        a = run_acc(X,Y)
        XL, YL, XU, YU = select_data2(XL_train, YL_train, XU_train, YU_train, feature_order, sel_num = i)
        a = run_acc2(XL, YL, XU, YU)
        acc_array[i] = a
#        print(a)
    return acc_array



def cal_many_acc_by_idx(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, idx_array):
#     print("idx_array", idx_array.shape)
    acc_array = np.zeros(idx_array.shape)
#     print("acc_array", acc_array.shape)
    for idx, i in enumerate(idx_array):
        X,Y = select_data(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, sel_num=i)
        a = run_acc(X,Y)
        acc_array[idx] = a
#        print(a)
    return acc_array


def cal_many_acc_by_idx2(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, idx_array):
#     print("idx_array", idx_array.shape)
    acc_array = np.zeros(idx_array.shape)
#     print("acc_array", acc_array.shape)
    for idx, i in enumerate(idx_array):
#        X,Y = select_data(XL_train, YL_train, XU_train, YU_train,\
#                           feature_order, sel_num=i)
#        a = run_acc(X,Y)
        XL, YL, XU, YU = select_data2(XL_train, YL_train, XU_train, YU_train, feature_order, sel_num = i)
        a = run_acc2(XL, YL, XU, YU)
        acc_array[idx] = a
#        print(a)
    return acc_array

 
def hist3d(y_label, x_label, z_value, \
           y_label_name = "y", x_label_name = "x", z_label_name = "z", output_filepath = "..\\result\\lsfs_featue_namuda.png"):

    """
    y -> row -> data.shape[0]
    x -> column -> data->shape[1]
    """
    
    column_names = x_label.copy()
    row_names = y_label.copy()



    data = z_value.copy()

    fig = plt.figure(figsize=(8, 5))
    ax = Axes3D(fig)

    lx= len(data[0])            # Work out matrix dimensions
    ly= len(data[:,0])
    xpos = np.arange(0,lx,1)    # Set up a mesh of positions
    ypos = np.arange(0,ly,1)
    xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = data.flatten()
    
    bar_colors = ['red', 'green', 'blue', 'aqua',
          'burlywood', 'cadetblue', 'chocolate', 'cornflowerblue',
          'crimson', 'darkcyan', 'darkgoldenrod', 'darkgreen',
          'purple', 'darkred', 'darkslateblue', 'darkviolet']
    
    colors = []
    
    for i in range(len(y_label)):
        for j in range(len(x_label)):
            colors.append(bar_colors[i])
            
    ax.bar3d(xpos,ypos,zpos, dx, dy, dz, color=colors, alpha = 0.6)

    #sh()
    ax.w_xaxis.set_ticklabels(x_label)
    ax.w_yaxis.set_ticklabels(y_label)
    ax.set_xlabel(x_label_name)
    ax.set_ylabel(y_label_name)
    ax.set_zlabel(z_label_name)

    #ax.w_xaxis.set_ticklabels(column_names)
    #ax.w_yaxis.set_ticklabels(row_names)

    # start, stop, step
    ticksx = np.arange(0.5, lx, 1)
    plt.xticks(ticksx, x_label)

    ticksy = np.arange(0.6, ly, 1)
    plt.yticks(ticksy, y_label)
    
    plt.savefig(output_filepath)
    print("save to ", output_filepath)
    plt.close()
    
#     plt.show()



def plot_array_like(array_1d, xlabel_name="number feature", ylabel_name="accuracy"):
    figsize = (8, 5)
    fig = plt.figure(figsize=figsize)

    print(array_1d)
    
    plt.plot(range(len(array_1d)), array_1d)

    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    plt.xlim(0, len(array_1d))
    # plt.ylim(0,1)
    plt.show()

#!usr/bin/python
# -*- coding:utf-8 -*-

from sSelect_FUN import *
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

output_file_name = "..\\result\\" + "sSelect_result" + "_" +  str(example_rate) + "_" + str(feature_rate) + "" + ".txt"

XL_train, YL_train, XU_train, YU_train  = read_data.get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                    unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)

# feature_order, time_dual, a = run_accuracy(lsfs, XL_train,YL_train,XU_train,YU_train, 10, output_file_name)

XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
# YL = read_data.label_n1_to_nc(YL)
# YU = read_data.label_n1_to_nc(YU)
# feature_order, time_dual = lsfs(XL, YL, XU, output_file_name="feature_order")

feature_order, time_dual = sSelect(XL, YL, XU, k = 10, theta = 10, namuda = 0.1, output_file_name=output_file_name)
# f_score = compute_Lr(XL, YL, XU, k = 10, theta = 10, namuda = 0.1)
# feature_order = np.argsort(f_score)

# feature_order = feature_order[::-1]

# np.random.shuffle(feature_order)

num_feature = len(feature_order)
#if num_feature > 300:
#    num_feature = 300

acc_array = evaluate.cal_many_acc(XL_train, YL_train, XU_train, YU_train,\
                           feature_order, num_feature = num_feature)


to_pickle("..\\result\\"+"sSelect_accuracy" + "_" +  str(example_rate) + "_" + str(feature_rate) + ".pkl", acc_array)


print(feature_order)
print("===================================================================")
print("===================================================================")
# print("accuracy : ", a)
print("time : ", time_dual)


evaluate.plot_array_like(acc_array, xlabel_name="number feature", ylabel_name="accuracy")




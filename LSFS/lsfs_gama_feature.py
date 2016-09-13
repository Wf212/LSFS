import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time
import sys
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import time
import sys

from LSFS_FUN import *
from LSFS_TEST import *
from read_data import *
from evaluate import *
from my_pickle import to_pickle



file_paths = [
    "..\\..\\data_selected\gene\\adenocarcinoma\\",
    "..\\..\\data_selected\\gene\\brain\\",
    "..\\..\\data_selected\\gene\\breast.2.class\\",
    "..\\..\\data_selected\\gene\\breast.3.class\\",
    "..\\..\\data_selected\\gene\\colon\\",
    "..\\..\\data_selected\\gene\\leukemia\\",
    "..\\..\\data_selected\\gene\\lymphoma\\",
    "..\\..\\data_selected\\gene\\nci\\",
    "..\\..\\data_selected\\gene\\prostate\\",
    "..\\..\\data_selected\\gene\\srbct\\"
]



cnt = 1
for file_path in file_paths:
    print(cnt, "    ",file_path)
    cnt += 1
    selected_data_file_name = "selected_data"
    # selected_feature_file_name = "selected_features"
    selected_cluster_name_file_name = "selected_cluster_names"

    unselected_data_file_name = "unselected_data"
    # unselected_feature_file_name = "unselected_features"
    unselected_cluster_name_file_name = "unselected_cluster_names"
    example_rate = 30
    feature_rate = 10

    output_file_name = "..\\result\\gama_feature\\" + "lsfs_gama_feature_result_" + file_path.split("\\")[-2] + "_" +  \
                                str(example_rate) + "_" + str(feature_rate)

    XL_train, YL_train, XU_train, YU_train  = get_data(file_path, selected_data_file_name, selected_cluster_name_file_name, \
                                        unselected_data_file_name, unselected_cluster_name_file_name, example_rate, feature_rate)

    # feature_order, time_dual, a = run_accuracy(lsfs, XL_train,YL_train,XU_train,YU_train, 10, output_file_name)

    XL, YL, XU, YU = XL_train.copy(), YL_train.copy(), XU_train.copy(), YU_train.copy()
    YL = read_data.label_n1_to_nc(YL)
    YU = read_data.label_n1_to_nc(YU)
    
    
    if XU.shape[1] > 300: 
        gama_array = np.array([10**i for i in range(-3, 4)])
        idx_array = np.array([50 * i + int(i==0) for i in range(7)])
        gama_f_array = compute_acc_diff_gama_fearture(XL_train, YL_train, XU_train, YU_train, gama_array, idx_array\
                                           , output_file_name="feature_order_tmp")
        
        to_pickle(output_file_name + ".pkl", gama_f_array ) 
        

        evaluate.hist3d(gama_array, idx_array, gama_f_array, \
                   y_label_name = "parameter $\gamma$", x_label_name = "# features", z_label_name = "ACC", \
                        output_filepath = output_file_name + ".png")






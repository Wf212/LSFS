# LSFS
Linear Semi-supervised Feature Selection

## 1
评估函数
evaluate/evaluate.py

## 2
生成一个二分类数据
gen_data/gen_data.py

读取数据的文件
gen_data/read_data.py

## 3
LSFS算法
LSFS/LSFS.py

LSFS算法相关函数
LSFS/LSFS_FUN.py

LSFS算法测试函数
LSFS/LSFS_TEST.py

数学函数L2_1norm
LSFS/math.py

解决问题
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
LSFS/EProjSimplex_new.py

## 4
计算pearson的函数
PRPC/my_math.py

PRPC算法
PRPC/PRPC.py

PRPC算法相关函数
PRPC/PRPC_FUN.py

## 5
laplacian_score算法
laplacian_score/laplacian_score.py

laplacian_score算法相关函数
laplacian_score/laplacian_score_FUN.py


laplacian_score算法测试函数
laplacian_score/laplacian_score_test.py


general_algorithm包含有knn函数
laplacian_score/general_algorithm.py

## 6
fisher_score算法的相关函数
Fisher/fisher_score.py

fisher_score算法
fisher_score_FUN.py

## 7
LSDF算法
LSDF/LSDF.py

LSDF算法相关函数
LSDF/LSDF_FUN.py


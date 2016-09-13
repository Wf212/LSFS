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

使用svm求分类的accuracy
LSFS/Baseline.py

用 对分类贡献前k大特征 进行分类时的accuracy
LSFS/feature_accuracy.py

LSFS的正则项取不同gama值时候，分类的accuracy
LSFS/get_gama.py

LSFS算法不同gama值，特征个数对应的accuracy，（三维柱状图）
LSFS/lsfs_gama_feature.py

LSFS算法（包括W的目标函数值）目标函数值随着迭代次数增加的变化
LSFS/objFun_iter.py

LSFS算法不同gama值，accuracy与特征个数的关系。（对accuracy由大到小排列取前K个来绘图，一个gama值一条曲线）
LSFS/plot_theta_feature.py


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



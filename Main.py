# -*- coding: utf-8 -*-
# @Time  : 2018/3/3 19:24
# @author: Hanrui Wang
nodeType = {"TERMINAL": 0, "NONTERMINAL": 1}
import os
from utils.dirtools import check_folder
import numpy as np
import utils.globalVars as gol
from gp import Consts
from gp import GSimpleGA, GTree
import Configurations as Configs
import ecoc_core.CallbackFuncs as CB
import ecoc_core.EvaluateMethods as EM
import ecoc_core.Initializator as Initializator
import ecoc_core.TreeMatrixConvertor as TMConvertor
from ecoc_core.ConnectClassifier import ConnectClassifier
from ecoc_core.OperationFuncs import *
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main_run():
    ##########################################
    # variables preparation
    ##########################################
    Initializator.init_gol()
    gol.set_val("final", 0)
    gol.set_val("aimFolder", Configs.aimFolder)
    gol.set_val("dataName", Configs.dataName)
    Initializator.init_all()  # 此时输出随机生成的种子库

    classes = gol.get_val("classes")
    maxDeap = gol.get_val("maxDeap")
    growMethod = gol.get_val("growMethod")
    generations = gol.get_val("generations")
    crossoverRate = gol.get_val("crossoverRate")
    mutationRate = gol.get_val("mutationRate")
    populationSize = gol.get_val("populationSize")
    freq_Stats = gol.get_val("freq_stats")
    columnBase = gol.get_val("columnBase")
    ##########################################

    genome = GTree.GTreeGP()  # 基因组为遗传规划的树的形式 初始化一个树形个体
    genome.setParams(max_depth=maxDeap, method=growMethod)
    genome.evaluator += EM.eval_func_fscore

    ga = GSimpleGA.GSimpleGA(genome)  # 将树形个体作为种群中的个体进行进化计算
    ga.setParams(gp_terminals=columnBase, gp_function_prefix="ternary")  # 种子是三进制串，运算符函数名是以ternary开头的，如ternary_Addition

    # modified by Liang Yifan
    # 收集所有操作运算函数名称
    Initializator.init_operStatistic(ga)

    # 初始化进化参数
    ga.setMinimax(Consts.minimaxType["maximize"])  # 个体的适应值越大越好
    ga.setGenerations(generations)  # 代数
    ga.setCrossoverRate(crossoverRate)  # 交叉率
    ga.setMutationRate(mutationRate)  # 突变率
    ga.setPopulationSize(populationSize)  # 种群的个体数
    ga.setElitismReplacement(1)  # 精英替代：1个

    # 每代需要执行的函数
    ga.stepCallback += CB.checkAncients_callback  # 检查初始代，输出相关信息
    ga.stepCallback += CB.logResultEveryGen_callback  # 输出特征向量，矩阵，要输出的结果的标题
    ga.stepCallback += CB.delogPopulation_callback  # 中间过程，包括每个个体的信息记录进文件中
    ga.stepCallback += CB.logMiddleInfo_callback  # 记录每代最优ACC Fscore进文件
    # ga.stepCallback += CB.printIndividuals_callback

    # 进化
    print("--------------------------begin----------------------------")
    ga(freq_stats=freq_Stats)
    print("---------------------------end-----------------------------")

    best = ga.bestIndividual()  # 从最终代找到最优个体

    # 将操作符统计输出到文件
    # modified by Liang Yifan
    operatorDF = gol.get_val("operatorDF")
    for node in best.nodes_list:
        if node.getType() == nodeType["NONTERMINAL"]:
            operator = str(node.node_data)
            operatorDF.loc[int(generations), operator]+=1
        else:
            continue
    Sum = operatorDF.apply(sum)
    operatorDF.loc[51] = Sum

    filedir = gol.get_val("root_path")
    filedir = os.path.join(filedir, 'Results/' + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("dataName") + "-v" + gol.get_val("version")+'/operatorStatistics/')
    check_folder(filedir)
    file = filedir + gol.get_val("aimFolder")+'operators.csv'
    operatorDF.to_csv(file, sep=',', header=True, index=True)

    # change the display_flag to display test labels and predict labels
    FinalMatrix, features_used_list = TMConvertor.getMatrixDirectly_and_feature(best)
    cc = ConnectClassifier(features_used_list, FinalMatrix)

    # 局部优化性能分析
    # modified by Liang Yifan
    score, accuracy = cc.TrainAndTestwithoutLocalImpr()
    finalFScore, finalAccuracy, txtText, new_ecocMatrix, new_features_used_list = cc.TrainAndTest()

    filedir = gol.get_val("root_path")
    filedir = os.path.join(filedir, 'Results/' + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("dataName") + "-v" + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("aimFolder"))
    check_folder(filedir)
    file = open(str(filedir)+"/localImprStatistics.txt",'w')
    file.write(str(FinalMatrix.shape[1])+','+str(new_ecocMatrix.shape[1])+'\n')
    file.write(str(score)+','+str(finalFScore))

    # euddist
    gol.set_val("final", 1)

    finalFScore, finalAccuracy = cc.FinalTrainAndTest()
    num_class = len(classes)
    num_cols = new_ecocMatrix.shape[1]
    _dist = euclidean_distances(new_ecocMatrix, new_ecocMatrix) / np.sqrt(num_cols)
    dist = np.sum(_dist) / 2 / (num_class * (num_class - 1))

    txtText.insert(len(txtText), "-------------test------------")
    txtText.insert(len(txtText), "accuracy: %f" % finalAccuracy)
    txtText.insert(len(txtText), "fscore: %f" % finalFScore)
    txtText.insert(len(txtText), "dist: %f" % dist)

    # =============================================================================
    print "-------------test------------"
    print "accuracy: %f" % finalAccuracy
    print "fscore: %f" % finalFScore
    print "dist: %f" % dist
    print new_features_used_list
    print new_ecocMatrix
    # =============================================================================


if __name__ == "__main__":
    main_run()

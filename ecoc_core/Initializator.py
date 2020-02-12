# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 10:25
# @Author: Hanrui Wang


import math
import numpy as np
import pandas as pd
import ecoc_core.RandomSeeds as RS
import Configurations as Configs
from preprocess import DataLoader, FeatureSelection
from utils import globalVars as gol

''' init all thing '''


def init_all():
    init_config()
    init_dataset()
    init_feature()
    init_maxColumn_and_maxDeap()
    init_columns()


def init_gol():
    gol._init()


''' load 'config' in gol '''


def init_config():
    dataName = gol.get_val("dataName")
    gol.set_val("n_jobs", Configs.n_jobs)
    gol.set_val("version", Configs.version)
    gol.set_val("testFile", Configs.root_path + "data/split/" + dataName + "_test.data")
    gol.set_val("trainFile", Configs.root_path + "data/split/" + dataName + "_train.data")
    gol.set_val("validFile", Configs.root_path + "data/split/" + dataName + "_validation.data")
    gol.set_val("generations", Configs.generations)
    gol.set_val("populationSize", Configs.populationSize)
    gol.set_val("freq_stats", Configs.freq_stats)
    gol.set_val("n_neighbors", Configs.n_neighbors)
    gol.set_val("crossoverRate", Configs.crossoverRate)
    gol.set_val("mutationRate", Configs.mutationRate)
    gol.set_val("growMethod", Configs.growMethod)
    gol.set_val("root_path", Configs.root_path)


def init_dataset():
    trainFile = gol.get_val("trainFile")
    validFile = gol.get_val("validFile")
    testFile = gol.get_val("testFile")
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = DataLoader.loadDataset(trainFile, validFile, testFile)
    classes = DataLoader.loadClasses(trainFile, validFile, testFile)
    gol.set_val("classes", classes)
    gol.set_val("Train_X", Train_X)
    gol.set_val("Train_Y", Train_Y)
    gol.set_val("Valid_X", Valid_X)
    gol.set_val("Valid_Y", Valid_Y)
    gol.set_val("Test_X", Test_X)
    gol.set_val("Test_Y", Test_Y)


def init_feature():
    # features
    Train_X = gol.get_val("Train_X")
    Train_Y = gol.get_val("Train_Y")
    trainFile = gol.get_val("trainFile")
    # the num of feature to be selected is 75 when feature space is larger than 75
    # f_n = Configs.feature_number
    feature_number = Train_X.shape[1]
    fea = FeatureSelection.select_features(trainFile, Train_X, Train_Y, feature_number)
    sel_features_backup = fea[0]
    sel_features = []
    for i in xrange(len(fea)):
        if i is 0:
            continue
        sel_features.append(fea[i])

    feature_method_index0 = dict((i, c) for i, c in enumerate(FeatureSelection.feature_method))
    feature_method_index = dict((c, i) for i, c in enumerate(FeatureSelection.feature_method))
    feature_method_index1 = {'svm25': '[0, 0]', 'svm50': '[0, 1]', 'svm75': '[0, -1]',
                             'bsswss25': '[1, 0]', 'bsswss50': '[1, 1]', 'bsswss75': '[1, -1]',
                             'forest25': '[-1, 0]', 'forest50': '[-1, 1]', 'forest75': '[-1, -1]'}
    feature_method_index2 = {'[0, 0]': 'svm25', '[0, 1]': 'svm50', '[0, -1]': 'svm75',
                             '[1, 0]': 'bsswss25', '[1, 1]': 'bsswss50', '[1, -1]': 'bsswss75',
                             '[-1, 0]': 'forest25', '[-1, 1]': 'forest50', '[-1, -1]': 'forest75'}
    gol.set_val("sel_features", sel_features)
    gol.set_val("feature_number", feature_number)
    gol.set_val("feature_method_index", feature_method_index)
    gol.set_val("feature_method_index0", feature_method_index0)
    gol.set_val("feature_method_index1", feature_method_index1)
    gol.set_val("feature_method_index2", feature_method_index2)
    gol.set_val("feature_method", FeatureSelection.feature_method)


''' Designed following the theory "2N_c" '''
def init_maxColumn_and_maxDeap():
    n_classes = float(len(gol.get_val("classes")))
    maxColumn = n_classes * 2
    maxDeap = np.ceil(math.log(maxColumn, 2)) + 1
    gol.set_val("maxColumn", int(maxColumn))
    gol.set_val("maxDeap", int(maxDeap))


def init_columns():
    gol.set_val("columnBase", RS.getColumns())
    print "columnBase:"
    print gol.get_val("columnBase")


# by Liang Yifan
# initialize--operator statistics
def init_operStatistic(ga_engine):
    operators = ga_engine.gp_collect_functions()
    operators.sort()
    gol.set_val("operators", ga_engine.gp_collect_functions())
    operators = gol.get_val("operators")
    generations = gol.get_val("generations")
    operatorDF = pd.DataFrame(np.random.randn(generations+1, len(operators)), columns=operators, dtype='int')
    operatorDF.loc[:, :] = 0
    gol.set_val("operatorDF", operatorDF)


def init_operNum():
    gol.set_val("operatorNum", {"ternary_Addition": 0, "ternary_HalfHalf": 0,
                                "ternary_LogicAnd": 0, "ternary_LogicOr": 0,
                                "ternary_Multiplication": 0, "ternary_OddEven": 0,
                                "ternary_Reverse": 0, "ternary_Subtraction": 0})


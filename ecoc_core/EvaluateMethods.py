# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 11:15
# @Author: Hanrui Wang


import TreeMatrixConvertor as TMConvertor
from ConnectClassifier import ConnectClassifier as CC
from DataComplexity import means_complexity
from DataComplexity import information_gain
from DataComplexity import information_gain_ratio
from DataComplexity import information_entropy

import numpy as np
from scipy.spatial import distance
from utils import globalVars as gol
from sklearn.metrics.pairwise import euclidean_distances


def eval_func_test(chromosome):
    """
    # calculate fscore
    """
    print type(chromosome)
    None


def eval_func_fscore(chromosome):
    """
    # calculate fscore
    """
    EcocMatrix, features_used_list = TMConvertor.getMatrixDirectly_and_feature(chromosome)

    cc = CC(features_used_list, EcocMatrix)
    fscore, accuracy, addDeleteList, new_ecocMatrix, new_features_used_list = cc.TrainAndTest()
    chromosome.hanrui = addDeleteList
    chromosome.new_ecocMatrix = new_ecocMatrix
    chromosome.new_features_used_list = new_features_used_list
    return fscore, accuracy


def eval_func_eucdist(chromosome):
    """
    # calculate avg_euclidean_dist of a individual
    """
    EcocMatrix, features_used_list = TMConvertor.getMatrixDirectly_and_feature(chromosome)
    classes = gol.get_val("classes")
    num_class = len(classes)
    num_cols = EcocMatrix.shape[1]
    _dist = euclidean_distances(EcocMatrix, EcocMatrix) / np.sqrt(num_cols)
    dist = np.sum(_dist) / 2 / (num_class * (num_class - 1))
    return dist


def eval_func_entropy(chromosome):
    """
    # Calculate the complexity named "means"
    # The data is all training set
    """
    Train_Y = gol.get_val("Train_Y")
    classes = gol.get_val("classes")
    EcocMatrix, features_used_list = TMConvertor.getMatrixDirectly_and_feature(chromosome)
    entropy = information_entropy(Train_Y, classes, EcocMatrix)
    return np.mean(entropy)


def eval_func_information_gain(chromosome):
    """
    # Calculate the information gain
    # The data is all training set
    """
    Train_Y = gol.get_val("Train_Y")
    classes = gol.get_val("classes")
    EcocMatrix, features_used_list = TMConvertor.getMatrixDirectly_and_feature(chromosome)
    infor_gain = information_gain(Train_Y, classes, EcocMatrix)
    return np.mean(infor_gain)


def eval_func_hamdist(chromosome):
    """
    # calculate hamdist of a individual
    """
    EcocMatrix, features_used_list = TMConvertor.getMatrixDirectly_and_feature(chromosome)
    classes = gol.get_val("classes")
    dist = 0
    for i in xrange(len(EcocMatrix)):
        for j in xrange(i + 1, len(EcocMatrix)):
            dist += distance.hamming(EcocMatrix[i], EcocMatrix[j])
    num = len(classes) * (len(classes) - 1) / 2
    dist /= num
    return dist

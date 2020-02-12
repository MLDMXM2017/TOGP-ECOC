# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 14:42
# @Author: Hanrui Wang


import numpy as np
from utils import globalVars as gol


def sameRows(Matrix):
    newMatrix = np.array(Matrix)
    for row in newMatrix:
        nums_same = (row == newMatrix).all(axis=1).tolist().count(True) - 1
        if nums_same > 0:
            return True
    return False


def zeroRow(Matrix):
    newMatrix = np.array(Matrix)
    zero_row = np.zeros(newMatrix.shape[1])
    return (zero_row == newMatrix).all(axis=1).any()


def zeroColumn(Matrix):
    # print Matrix
    newMatrix = np.array(Matrix).transpose()
    # print newMatrix
    zero_column = np.zeros(newMatrix.shape[1])
    deletes = np.where((zero_column == newMatrix).all(axis=1) == True)[0].tolist()
    return deletes


def onlyOneColumn(Matrix):
    # check if there exists a column that is lack of 1 or -1
    # the return value of zero column will be False
    newMatrix = np.array(Matrix)
    deletes = []
    for i in xrange(newMatrix.shape[1]):
        col = newMatrix[:, i]
        if (-1 not in col) | (1 not in col):
            deletes.append(i)
        # # check 1
        # one_exist_flag = False
        # ng_one_exist_flag = False
        # for j in xrange(len(col)):
        #     if (col[j] == 1):
        #         one_exist_flag = True
        #     elif (col[j] == -1):
        #         ng_one_exist_flag = True
        #
        # if one_exist_flag == False and ng_one_exist_flag == True:
        #     deletes.append(i)
        # elif one_exist_flag == True and ng_one_exist_flag == False:
        #     deletes.append(i)
    return deletes


def sameColumns(Matrix, feature_list):
    newMatrix = np.array(Matrix)

    def sameRows_for_feature(newMatrix, feature_list):
        deletes = set()
        for rowi in xrange(newMatrix.shape[0]):
            for rowj in range(rowi + 1, newMatrix.shape[0]):
                if (newMatrix[rowi, :] == newMatrix[rowj, :]).all():
                    # To valid whether the feature of col_i and col_j are same
                    if feature_list[rowi] == feature_list[rowj]:  deletes.add(rowj)
        return list(deletes)

    return sameRows_for_feature(np.transpose(newMatrix), feature_list)


def opstColumns(Matrix, feature_list):
    newMatrix = np.array(Matrix)
    deletes = set()
    for i in xrange(newMatrix.shape[1]):
        newMatrix[:, i] = -1 * newMatrix[:, i]
        for e in sameColumns(newMatrix, feature_list):
            deletes.add(e)
        newMatrix[:, i] = -1 * newMatrix[:, i]
    return list(deletes)


def tooLittleColumn(Matrix):
    newMatrix = np.array(Matrix)
    if newMatrix.shape[1] < newMatrix.shape[0] - 1:
        return True
    else:
        return False


def tooMuchColumn(Matrix):
    newMatrix = np.array(Matrix)
    if newMatrix.shape[1] > gol.get_val("maxColumn"):
        return True
    else:
        return False

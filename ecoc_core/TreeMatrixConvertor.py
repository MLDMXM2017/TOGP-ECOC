# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 14:53
# @Author: Hanrui Wang
# @Target:


import copy
import numpy as np
import LegalityCheckers as LC
from gp import GTree
from gp.GenomeBase import GTreeBase
from utils import globalVars as gol
from ecoc_core.OperationFuncs import *

nodeType = {"TERMINAL": 0, "NONTERMINAL": 1}

'''
# convert Tree to EcocMatrix, 
# and then use LegalityCheckers to delete illegal columns
'''


def getMatrixDirectly_and_feature(ind):
    cloned = ind.clone()

    arrays = []
    feature = []

    features = gol.get_val("feature_method")
    classes = gol.get_val("classes")
    MaxDeapth = gol.get_val("maxDeap")

    for i in xrange(len(cloned.nodes_list)):
        if cloned.nodes_list[i].getType() == nodeType["NONTERMINAL"]:
            for j in xrange(0, len(classes)):
                locals()[classes[j]] = classes[j]
            NewInd = GTree.GTreeGP()
            NewInd.setRoot(cloned.nodes_list[i])
            array = eval(NewInd.getCompiledCode())
            arrays.append(list(array))
        else:
            arrays.append(eval(cloned.nodes_list[i].getData()))

    for i in xrange(len(arrays)):
        for j in xrange(len(features)):
            if features[j] in arrays[i]:
                feature.append(features[j])
                arrays[i].remove(features[j])

    ecocMatrix = np.array(arrays).transpose()
    feature = np.array(feature)

    #############################
    #      hanrui's modifier    #
    #############################
    # 1.There being a column that is lack of 1 or -1
    deletes = LC.onlyOneColumn(ecocMatrix)
    ecocMatrix = np.delete(ecocMatrix, deletes, axis=1)
    feature = np.delete(feature, deletes, axis=0)

    # 2.Two columns having the same numbers
    deletes = LC.sameColumns(ecocMatrix, feature)
    ecocMatrix = np.delete(ecocMatrix, deletes, axis=1)
    feature = np.delete(feature, deletes, axis=0)

    # 3.Two columns having the opposite numbers
    deletes = LC.opstColumns(ecocMatrix, feature)
    ecocMatrix = np.delete(ecocMatrix, deletes, axis=1)
    feature = np.delete(feature, deletes, axis=0)
    #############################
    return ecocMatrix, feature


def getMatrixDirectly(ind):
    ind_info = getMatrixDirectly_and_feature(ind)
    return ind_info[0]

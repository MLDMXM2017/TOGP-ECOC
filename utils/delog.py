#!/usr/bin/python2
# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 14:11
# @Author: Hanrui Wang


import os
import sys
import numpy as np
import globalVars as gol
import ecoc_core.Initializator as initializator
from utils.dirtools import check_folder

nodeType = {"TERMINAL": 0, "NONTERMINAL": 1}

def deprint(tag, value):
    print "===%s====start====" % tag
    print value
    print "===%s=====end=====" % tag


def deprint_string(string):
    print string


def decache(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def logPopulations(genid, pop, operatorDF):

    filedir = gol.get_val("root_path")
    filedir = os.path.join(filedir, 'Results/' + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("dataName") + "-v" + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("aimFolder"))
    check_folder(filedir)
    filedir = os.path.join(filedir, "Gen." + str(genid))

    f = file(filedir, 'w+')

    #######################################################################
    #  by Liang Yifan
    #  operators statistics  to csv
    bestind = pop.bestRaw()
    for node in bestind.nodes_list:
        if node.getType() == nodeType["NONTERMINAL"]:
            operator = str(node.node_data)
            operatorDF.loc[int(genid), operator]+=1
        else:
            continue
    ######################################################################

    i = 1
    for ind in pop:
        #######################################################################
        #  by Liang Yifan
        initializator.init_operNum()
        operatorNum = gol.get_val("operatorNum")

        for node in ind.nodes_list:
            if node.getType() == nodeType["NONTERMINAL"]:
                operator = str(node.node_data)
                operatorNum[operator] += 1
            else:
                continue

        statistic = "Operator statistics:\n"
        for item in operatorNum:
            if operatorNum.get(str(item)) == 0:
                continue
            else:
                statistic += item + ":" + str(operatorNum.get(str(item))) + "\n"
        statistic += "\n"
        #######################################################################

        Matrix = ind.new_ecocMatrix
        Matrix = np.array(Matrix)
        feature_list = ind.new_features_used_list

        f.write("##############\n")
        f.write(" Individual " + str(i) + "\n")
        f.write("##############\n")

        # every gen every individual operator statistics  by Liang Yifan
        f.write(statistic)

        f.write(str(feature_list) + '\n')
        f.write(str(Matrix) + '\n')

        f.write("F-Score: " + str(ind.fscore) + "\t\tAccuracy: " + str(ind.accuracy) + "\n")
        f.write('\n')

        f.write('-' * 37 + 'The train and valid Process' + '-' * 36 + '\n')
        for text in ind.hanrui:
            f.write(str(text))
            f.write('\n')
        f.write('\n\n\n\n')

        i += 1

    f.close()


def logMiddle(genid, fitness, filename, mode):
    filedir = gol.get_val("root_path")
    filedir = os.path.join(filedir, 'Results/' + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("dataName") + "-v" + gol.get_val("version"))
    filedir = os.path.join(filedir, gol.get_val("aimFolder"))
    check_folder(filedir)
    filedir = os.path.join(filedir, filename)

    f = file(filedir, 'a')
    f.write(str(genid))
    f.write(":")
    f.write('\n')
    if mode == "acc":
        f.write("best Accuracy:")
    elif mode == "fscore":
        f.write("best Fscore:")
    f.write(str(fitness))
    f.write('\n')
    f.write('\n')
    f.close()

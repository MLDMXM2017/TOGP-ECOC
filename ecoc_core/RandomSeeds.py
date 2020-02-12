# -*- coding: utf-8 -*-
# @Time  : 2018/3/9 17:01
# @Author: Hanrui Wang
# @Target: Generate some random seeds to make up a columnBase

import random
from utils import globalVars as gol


def getColumns():
    classes = gol.get_val("classes")
    columnBase = []
    numBase = len(classes) * 3
    num = 0
    while num < numBase:
        while True:
            column = []
            featureNumList = []
            featureNumList.append(random.randint(-1, 1))
            featureNumList.append(random.randint(-1, 1))
            feature_num = str(featureNumList)

            feature_method_index2 = gol.get_val("feature_method_index2")
            column.append(feature_method_index2[feature_num]) 
            # 随机生成列
            for i in xrange(len(classes)):
                column.append(random.randint(-1, 1))
            if (-1 not in column) | (1 not in column):
                continue
            num += 1
            columnBase.append(str(column))
            break
    return columnBase

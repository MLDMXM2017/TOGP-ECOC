# -*- coding: utf-8 -*-
# @Time  : 2018/3/1 15:53
# @author: Hanrui Wang
# modifid by Liang Yifan
import copy
from utils import globalVars as gol
import preprocess.FeatureSelection as fs

feature_method_index = dict((i, c) for i, c in enumerate(fs.feature_method))

#    0 1 -1
#  0
#  1
# -1
logicTable_or = [[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, -1]]

logicTable_and = [[0, 0, -1],
                     [0, 1, -1],
                     [-1, -1, -1]]

logicTable_reverse = [[0, -1, 1],
                      [-1, 1, 0],
                      [1, 0, -1]]


def refreshColumn(column, location, num):

    if location is -1:
        return column

    if (column[location] is 1) & (num is 1):
        column[location] = -1
        return refreshColumn(column, location - 1, 1)
    elif (column[location] is -1) & (num is -1):
        column[location] = 1
        return refreshColumn(column, location - 1, -1)
    else:
        column[location] += num
        return column


'''
a. Addition
-1 + -1 = -11，-1 + 0 = -1，  -1 + 1 = 0
 0 + -1 = -1   0 + 0 = 0，    0 + 1 = 1
 1 + -1 = 0，  1 + 0 = 1，    1 + 1 = 1-1
'''
def ternary_Addition(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)

    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])  # 得到的是编码
    fea2 = eval(feature_method_index1[c2[0]])  # 得到的是编码

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])

    fea = []

    location = len(fea1) - 1
    while True:
        if (fea1[location] is 1) & (fea2[location] is 1):  # 1+1=-1
            fea.append(-1)
            fea2 = refreshColumn(fea2, location, 1)
        elif (fea1[location] is -1) & (fea2[location] is -1):  # -1+-1=1
            fea.append(1)
            fea2 = refreshColumn(fea2, location, -1)
        else:
            fea.append(fea1[location] + fea2[location])  # 正常加法
        location -= 1

        if location is -1:
            break

    fea.reverse()

    fea = feature_method_index2[str(fea)]  # 得到的是方法名称而非编码

    # 三进制运算
    del c1[0]
    del c2[0]

    column3 = []
    location = len(c1) - 1
    while True:
        if (c1[location] is 1) & (c2[location] is 1):
            column3.append(-1)
            c2 = refreshColumn(c2, location, 1)

        elif (c1[location] is -1) & (c2[location] is -1):
            column3.append(1)
            c2 = refreshColumn(c2, location, -1)
        else:
            column3.append(c1[location] + c2[location])
        location -= 1

        if location is -1:
            break

    column3.reverse()
    column3.insert(0, fea)

    return column3


'''
b. Subtraction 
-1 – -1 = 0，  -1 – 0 = -1，   -1 – 1 = -11
 0 – -1 = 1，   0 – 0 = 0，    0 – 1 = -1 
 1 – -1 = 1-1， 1 – 0 = 1，    1 – 1 = 0
'''
def ternary_Subtraction(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)

    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])
    fea2 = eval(feature_method_index1[c2[0]])

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])

    fea = []

    location = len(fea1) - 1
    while True:
        if (fea1[location] is 1) & (fea2[location] is 1):
            fea.append(-1)
            fea2 = refreshColumn(fea2, location, 1)
        elif (fea1[location] is -1) & (fea2[location] is -1):
            fea.append(1)
            fea2 = refreshColumn(fea2, location, -1)
        else:
            fea.append(fea1[location] + fea2[location])
        location -= 1

        if location is -1:
            break
    fea.reverse()
    fea = feature_method_index2[str(fea)]

    # 三进制运算
    del c1[0]
    del c2[0]

    for i in xrange(len(c2)):
        if i is 0:
            continue
        c2[i] *= -1

    column3 = []
    location = len(c1) - 1
    while True:
        if (c1[location] is 1) & (c2[location] is 1):
            column3.append(-1)
            c2 = refreshColumn(c2, location, 1)
        elif (c1[location] is -1) & (c2[location] is -1):
            column3.append(1)
            c2 = refreshColumn(c2, location, -1)
        else:
            column3.append(c1[location] + c2[location])
        location -= 1

        if location is -1:
            break
    column3.reverse()

    column3.insert(0, fea)

    return column3


'''
c. Multiplication
-1 * -1 = 1，   -1 * 0 = 0，   -1 * 1 = -1
 0 * -1 = 0，   0 * 0 = 0，    0 * 1 = 0
 1 * -1 = -1，   1 * 0 = 0，    1 * 1 = 1
 '''
def ternary_Multiplication(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)
    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])
    fea2 = eval(feature_method_index1[c2[0]])

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])

    fea = []
    for location in xrange(len(fea1)):
        fea.append(fea1[location] * fea2[location])
    fea = feature_method_index2[str(fea)]

    # 三进制运算
    del c1[0]
    del c2[0]

    column3 = []
    for location in xrange(len(c1)):
        column3.append(c1[location] * c2[location])

    column3.insert(0, fea)

    return column3


'''
d. or
-1∨-1 = -1，  -1∨0 = 0，   -1∨1 = 1
 0∨-1 = 0，   0∨0 = 0，    0∨1 = 1
 1∨-1 = 1，   1∨0 = 1，    1∨1 = 1
'''
def ternary_LogicOr(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)
    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])
    fea2 = eval(feature_method_index1[c2[0]])

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])
    fea = []
    for location in xrange(len(fea1)):
        if fea1[location] is -1:
            fea1[location] = 2
        if fea2[location] is -1:
            fea2[location] = 2
        fea.append(logicTable_or[fea1[location]][fea2[location]])
    fea = feature_method_index2[str(fea)]

    # 三进制运算
    del c1[0]
    del c2[0]

    column3 = []
    for location in xrange(len(c1)):
        if c1[location] is -1:
            c1[location] = 2
        if c2[location] is -1:
            c2[location] = 2
        column3.append(logicTable_or[c1[location]][c2[location]])

    column3.insert(0, fea)

    return column3


'''
e. and
-1∧-1 = -1，  -1∧0 = -1，   -1∧1 = -1
 0∧-1 = -1，  0∧0 = 0，    0∧1 = 0
 1∧-1 = -1，   1∧0 = 0，    1∧1 = 1
'''
def ternary_LogicAnd(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)

    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])
    fea2 = eval(feature_method_index1[c2[0]])

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])
    fea = []
    for location in xrange(len(fea1)):
        if fea1[location] is -1:
            fea1[location] = 2
        if fea2[location] is -1:
            fea2[location] = 2
        fea.append(logicTable_and[fea1[location]][fea2[location]])
    fea = feature_method_index2[str(fea)]

    # 三进制运算
    del c1[0]
    del c2[0]

    column3 = []
    for location in xrange(len(c1)):
        if c1[location] is -1:
            c1[location] = 2
        if c2[location] is -1:
            c2[location] = 2
        column3.append(logicTable_and[c1[location]][c2[location]])

    column3.insert(0, fea)

    return column3

#####################
# by Liang Yifan
# New operators
#####################
'''
C[k] = CLeft[i](k = i%2==0) C[k] = CRight[j](k = j%2!=0)
'''
def ternary_OddEven(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)

    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])  # 得到的是编码
    fea2 = eval(feature_method_index1[c2[0]])  # 得到的是编码

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])

    fea = []
    for location in xrange(len(fea1)):
        if location % 2 is 0:
            fea.append(fea1[location])
        else:
            fea.append(fea2[location])

    column3 = []
    for location in xrange(1, len(column1)):
        if (location+1) % 2 is 0:
            column3.append(c1[location])
        else:
            column3.append(c2[location])

    fea = feature_method_index2[str(fea)]
    column3.insert(0, fea)

    return column3


'''
C[k] = CLeft[i](0 <= k = i < L/2) C[k] = CRight[j](L/2 <= k = j < L)
'''
def ternary_HalfHalf(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)

    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])  # 得到的是编码
    fea2 = eval(feature_method_index1[c2[0]])  # 得到的是编码

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])

    fea = []

    fea.append(fea1[0])
    fea.append(fea1[1])

    column3 = []
    c1.insert(0, 0)
    c2.insert(0, 0)
    for location in xrange(2, len(c1)):
        if location < len(c1)/2:
            column3.append(c1[location])
        else:
            column3.append(c2[location])

    fea = feature_method_index2[str(fea)]
    column3.insert(0, fea)

    return column3


'''
C[i] = {-1, 0, 1} - CLeft[i] - CRight[j](0 <= i < L)
'''
def ternary_Reverse(column1, column2):

    c1 = copy.copy(column1)
    c2 = copy.copy(column2)
    # 特征选择方法二进制运算
    feature_method_index1 = gol.get_val("feature_method_index1")
    feature_method_index2 = gol.get_val("feature_method_index2")
    fea1 = eval(feature_method_index1[c1[0]])
    fea2 = eval(feature_method_index1[c2[0]])

    for i in xrange(len(fea1)):
        fea1[i] = int(fea1[i])
        fea2[i] = int(fea2[i])
    fea = []
    for location in xrange(len(fea1)):
        if fea1[location] is -1:
            fea1[location] = 2
        if fea2[location] is -1:
            fea2[location] = 2
        fea.append(logicTable_reverse[fea1[location]][fea2[location]])
    fea = feature_method_index2[str(fea)]

    # 三进制运算
    del c1[0]
    del c2[0]

    column3 = []
    for location in xrange(len(c1)):
        if c1[location] is -1:
            c1[location] = 2
        if c2[location] is -1:
            c2[location] = 2
        column3.append(logicTable_reverse[c1[location]][c2[location]])

    column3.insert(0, fea)

    return column3
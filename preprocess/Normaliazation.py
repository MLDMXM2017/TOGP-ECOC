# -*- coding: utf-8 -*-
# @author Liang Yifan
'''
# to normalize raw data
# Raw data should be:
#  1. a sample each column
#  2. digital features
#  3. fileName :  "xxx.data"
'''


from numpy import double, loadtxt
from sklearn import preprocessing
import numpy as np


def loadFeatures(filename):
    dataSet = np.array(loadtxt(filename,skiprows=1,dtype=double,ndmin=2,delimiter=','))
    return np.ndarray.tolist(np.transpose(dataSet))


def loadStrLabel(filename):
    file = open(filename, "r")
    line = file.readline()
    line = line.strip()  # delete the last character
    labels = line.split(',')
    file.close()
    return labels


def Normalization(dName):
    dname = "../data/raw/"+dName+".data"
    X = loadFeatures(dname)
    y = loadStrLabel(dname)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    return X,y


def dataIO(fName, X, y):
    X = np.ndarray.tolist(np.transpose(np.array(X)))
    tfile = open("../data/normalized/"+fName, "w+")
    tfile.write(",".join(list(map(str,y)))+"\n")
    for x_row in X:
        line = ",".join(list(map(str,x_row)))
        tfile.write(line+"\n")
    tfile.close()


#主函数
if __name__ == "__main__":

    datasets = ["abalone", "balance", "cmc", "dermatology", "ecoli", "glass",
                "iris", "Leukemia1", "Leukemia2", "mfeatzer", "optdigits","pendigits",
                "sat", "segment", "thyroid", "vehicle", "vertebral", "waveforms",
                "wine", "yeast", "zoo", "cifar_10", "fashion_MNIST", "JAFFE", "SAMM", "Yale"]
                
    for dName in datasets:
        print "===="+dName+" begin"+"===="
        print "Normalizing..."
        X, y = Normalization(dName)
        print "datafile io..."
        dataIO(dName+".data", X, y)
        print "===="+dName+" end"+"===="
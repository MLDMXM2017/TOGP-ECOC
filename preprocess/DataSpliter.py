# -*- coding: utf-8 -*-
'''
# to split raw data into training set and testing set

# Notice!!!!!
# Raw data should be:
#  1. a sample each column
#  2. digital features
#  3. fileName :  "xxx.data"
'''




from numpy import double, fromfile, loadtxt
from sklearn.model_selection import train_test_split
import numpy as np


def loadFeatures(filename):
    dataSet = np.array(loadtxt(filename,skiprows=1,dtype=double,ndmin=2,delimiter=','))
    return  np.ndarray.tolist(np.transpose(dataSet))


def loadStrLabel(filename):
    file = open(filename, "r")
    line = file.readline()
    line = line.strip()  # delete the last character
    labels = line.split(',')
    file.close()
    return labels

def spliterOne(dName):
    dname = "../data/normalized/"+dName+".data"
    X = loadFeatures(dname)
    y = loadStrLabel(dname)
    # divide into training set and testing set , 5 : 5
    X_train, X_t_v, y_train, y_t_v = train_test_split(X, y, test_size=0.5,
                                                      random_state=1)
    # divide into validation set and testing set , 5 : 5
    X_validation, X_test, y_validation, y_test = train_test_split(X_t_v, y_t_v, test_size=0.5,
                                                                  random_state=1)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def dataIO(fName, x, y):
    x = np.ndarray.tolist(np.transpose(np.array(x)))
    tfile = open("../data/splited/"+fName, "w+")
    tfile.write(",".join(list(map(str,y)))+"\n")
    for x_row in x:
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
        print "spliting..."
        X_train, X_validation, X_test, y_train, y_validation, y_test = spliterOne(dName)
        print "train io..."
        dataIO(dName+"_train.data",X_train,y_train)
        print "validation io..."
        dataIO(dName+"_validation.data",X_validation,y_validation)
        print "testing io..."
        dataIO(dName+"_test.data",X_test,y_test)
        print "===="+dName+" end"+"===="
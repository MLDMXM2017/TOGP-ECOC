# -*- coding: utf-8 -*-
# @Author Liang Yifan
import os
import numpy as np
from ecoc_core import Initializator

from utils.dirtools import check_folder
import Configurations as Configs


def localImproAnalysis(dataset, filecol, filefscore):

    columnChanges = []
    fscoreChanges = []
    for i in xrange(1, 11):
        f = file("../Results/"+Configs.version+'/'+str(dataset)+"-v"+Configs.version+"/ex"+str(i)+'/localImprStatistics.txt','r')
        columns = f.readline()
        fscores = f.readline()
        column = (columns.strip('\n')).split(',')
        column = [float(x) for x in column]
        fscore = (fscores.strip('\n')).split(',')
        fscore = [float(x) for x in fscore]
        columnChanges.append(column)
        fscoreChanges.append(fscore)
    columnChanges = np.array(columnChanges)
    fscoreChanges = np.array(fscoreChanges)
    columnChanges = np.mean(columnChanges, axis=0)
    fscoreChanges = np.mean(fscoreChanges, axis=0)
    filecol.write(str(dataset)+',')
    filecol.write(str(columnChanges[0])+',')
    filecol.write(str(columnChanges[1])+',\n')

    filefscore.write(str(dataset)+',')
    filefscore.write(str(fscoreChanges[0])+',')
    filefscore.write(str(fscoreChanges[1])+',\n')


if __name__ == "__main__":

    datasets = ["abalone", "balance", "cmc", "dermatology", "ecoli", "glass",
                "iris", "Leukemia1", "Leukemia2", "mfeatzer", "optdigits","pendigits",
                "sat", "segment", "thyroid", "vehicle", "vertebral", "waveforms",
                "wine", "yeast", "zoo", "cifar_10", "fashion_MNIST", "JAFFE", "SAMM", "Yale"]

    filedir = '../'
    filedir = os.path.join(filedir, 'Results/' + str(Configs.version)+'/')
    filedir = os.path.join(filedir, 'Analysis/localImprovement')
    check_folder(filedir)
    filecol = open(filedir + '/localImprCol.csv','a')
    filefscore = open(filedir + '/localImprFScore.csv','a')
    filecol.truncate()
    filefscore.truncate()
    for dataset in datasets:
        localImproAnalysis(dataset, filecol, filefscore)

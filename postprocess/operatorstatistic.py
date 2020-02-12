# -*- coding: utf-8 -*-
# @Author Liang Yifan
import os
import numpy as np
import pandas as pd
import utils.globalVars as gol
from ecoc_core import Initializator
from utils.dirtools import check_folder
import Configurations as Configs


def avgoperators():
    
    datasets = ["abalone", "balance", "cmc", "dermatology", "ecoli", "glass",
                "iris", "Leukemia1", "Leukemia2", "mfeatzer", "optdigits","pendigits",
                "sat", "segment", "thyroid", "vehicle", "vertebral", "waveforms",
                "wine", "yeast", "zoo", "cifar_10", "fashion_MNIST", "JAFFE", "SAMM", "Yale"]


    operators = ["ternary_OddEven","ternary_Addition","ternary_Subtraction","ternary_HalfHalf","ternary_LogicOr","ternary_Multiplication","ternary_Reverse","ternary_LogicAnd"]
    average = pd.DataFrame(columns=operators, index=datasets, dtype='float')
    summation = pd.DataFrame(columns=operators, index=datasets, dtype='float')
    Initializator.init_gol()
    gol.set_val("aimFolder", Configs.aimFolder)
    gol.set_val("version", Configs.version)
    filedir = '../'
    filedir = os.path.join(filedir, 'Results/' + gol.get_val("version")+'/')

    for d in xrange(len(datasets)):
        filedir1 = os.path.join(filedir, str(datasets[d]) + "-v" + gol.get_val("version")+'/operatorStatistics/')
        check_folder(filedir1)
        statistic = []
        for i in xrange(10):
            file = filedir1 + 'ex' + str(i+1) + 'operators.csv'
            df = pd.read_csv(file, dtype='int', index_col=0)
            statistic.append(df.loc[50])  
        statistic = np.array(statistic)  
        avg = np.mean(statistic, axis=0)
        average.loc[datasets[d]] = avg
        sum = np.sum(statistic, axis=0)
        summation.loc[datasets[d]] = sum
    filedir2 = '.' + str(Configs.root_path)
    filedir2 = os.path.join(filedir2, 'Results/' + gol.get_val("version")+'/Analysis/operators/')
    check_folder(filedir2)
    average.to_csv(filedir2+'operatorsAvg.csv', header=True, index=True)
    summation.to_csv(filedir2+'operatorsSum.csv', header=True, index=True)


if __name__ == "__main__":
    avgoperators()

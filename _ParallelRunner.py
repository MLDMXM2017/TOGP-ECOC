# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:18:46 2017

@author: Shone
"""

import multiprocessing as mul
import os
import time
import shutil
import sys

from utils.stack import Stack

import Configurations as Configs
from ecoc_core.OperationFuncs import *
from postprocess import Controller
from utils.dirtools import check_folder, del_dir_tree


def _agp_main_runner(dataName, aimFolder):
    std = sys.stdout
    try:
        Configs.dataName = dataName
        Configs.aimFolder = aimFolder
        _rootp = Configs.root_path
        _rootp = os.path.join(_rootp, 'Results/' + Configs.version)
        _rootp = os.path.join(_rootp, Configs.dataName + "-v" + Configs.version)
        _rootp = os.path.join(_rootp, Configs.aimFolder)
        del_dir_tree(_rootp)    # delete former results
        check_folder(_rootp)
        _res = "/Result"
        out = open(_rootp + _res ,'w+')
        sys.stdout = out
        sys.stderr = out
        import Main as main_module
        main_module.main_run()
    except Exception as e:
        _err = "/Error"
        err = open(_rootp + _err, 'w+')
        err.write(e.message)
        err.flush()
    finally:
        sys.stdout = std


if __name__ == "__main__":

    # 用于测试数据集
    # datasets = ['iris']

    # 新增数据集
    # datasets = ["pendigits", "fashion_MINIST", "jaffedbase", "cifar_10"]

    datasets = ["abalone", "balance", "cmc", "dermatology", "ecoli",
                "glass", "iris", "Leukemia_3c", "MLL", "mfeatzer",
                "optdigits", "sat", "segment", "thyroid", "vehicle",
                "vertebral", "waveforms", "wine", "yeast", "zoo"]

    # 初始化队列
    experiments = list()
    for dataName in datasets:
        s = Stack()
        for i in xrange(10):
            aimFolder = "ex" + str(10 - i)
            s.push((dataName, aimFolder))
        experiments.append(s)

    # 第一批实验
    p_list = list()
    for exp in experiments:
        (dataName, aimFolder) = exp.pop()
        p = mul.Process(target=_agp_main_runner, args=(dataName, aimFolder))
        print dataName + " " + aimFolder + " begin"
        p.start()
        p_list.append(p)

    while True:
        time.sleep(2)
        index_remove = list()
        for i in xrange(len(p_list)):
            # do something when a process is over.
            if not p_list[i].is_alive():
                if not experiments[i].isEmpty():
                    (dataName, aimFolder) = experiments[i].pop()
                    p = mul.Process(target=_agp_main_runner, args=(dataName, aimFolder))
                    print dataName + " " + aimFolder + " begin"
                    p.start()
                    p_list[i] = p
                else:
                    index_remove.append(i)
        if len(index_remove) == 0:
            continue
        index_remove.reverse()
        for i in index_remove:
            experiments.remove(experiments[i])
            p_list.remove(p_list[i])
        if len(p_list) == 0:
            break

    # parse result
    genid = 5
    genids = [1]
    while genid <= Configs.generations:
        genids.append(genid)
        genid = genid + 5

    _rootp = Configs.root_path
    _rootp = os.path.join(_rootp, 'Results/' + Configs.version)
    print "result parse begin"
    Controller._parseRunner(_rootp, genids)
    print "result parse end"

    # copy xlsm template
    _vp = Configs.root_path + "Results/"+Configs.version+\
            "/a_s"+Configs.version+"/result-"+Configs.version+".xlsm"
    # if not os.path.isfile(_vp):  # 如果不存在这个文件
    shutil.copyfile(Configs.root_path + "dataTemplate/rTemplate.xlsm", _vp)

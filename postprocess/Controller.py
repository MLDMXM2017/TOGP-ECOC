# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:51:46 2017

@author: Shone
"""
import shutil
import Configurations as Configs
import Parsers

from utils.stack import Stack


def _parseRunner(root, genids):
    ps = Stack()
    ps.push(Parsers.ParserAHD())
    ps.push(Parsers.ParserColumn())
    ps.push(Parsers.ParserFitness())
    ps.push(Parsers.ParserTestAcc())
    ps.push(Parsers.ParserTestFscore())
    ps.push(Parsers.ParserTrainAcc())
    ps.push(Parsers.ParserTrainFscore())

    while not ps.isEmpty():
        ps.pop().parse(root, genids)


if __name__ == '__main__':
    genids = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
              55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    root = '../Results/4.0/'
    _parseRunner(root, genids)
    _vp = "../Results/"+Configs.version+'/'+ \
          "/a_s"+Configs.version+"/result-"+Configs.version+".xlsm"
    shutil.copyfile("../dataTemplate/rTemplate.xlsm", _vp)
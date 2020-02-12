# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 14:03
# @Author: Hanrui Wang

from ecoc_core.OperationFuncs import *
import LegalityCheckers as LCheckers
import TreeMatrixConvertor as TMConvertor
from ConnectClassifier import ConnectClassifier as CC
import sys
import copy
import numpy as np
from gp import Util, GTree, Consts
from utils import delog
from utils import globalVars as gol
import Configurations as Configs
import itertools

# log the best accuracy of every generation into file AaccPerGen
# log the best fscore in every generation into file AfscorePerGen
def logMiddleInfo_callback(gp_engine):
    sys.stdout.write("logMiddleInfo...")
    genid = gp_engine.getCurrentGeneration()
    i = genid+1
    best = gp_engine.bestIndividual()
    FinalMatrix = best.new_ecocMatrix
    features_used_list = best.new_features_used_list

    cc = CC(features_used_list, FinalMatrix)
    finalScore, finalAccuracy = cc.FinalTrainAndTest()

    delog.logMiddle(i, finalAccuracy, "AaccPerGen", "acc")
    delog.logMiddle(i, finalScore, "AfscorePerGen", "fscore")

    sys.stdout.write("over\n")
    sys.stdout.flush()


def delogPopulation_callback(gp_engine):
    operatorDF = gol.get_val("operatorDF")
    pop = gp_engine.getPopulation()
    genid = gp_engine.getCurrentGeneration()
    delog.logPopulations(genid, pop, operatorDF)
    gol.set_val("operatorDF", operatorDF)


def logResultEveryGen_callback(gp_engine):
    # if gp_engine.getCurrentGeneration() == 0:
    print "=" * 65
    np.set_printoptions(threshold=None)
    # do in every generation
    ind = gp_engine.getPopulation().bestRaw()
    bestMatrix, feature_list = TMConvertor.getMatrixDirectly_and_feature(ind)
    # feature_method_index = gol.get_val("feature_method_index")
    # feature_index_list = list(feature_method_index[method] for method in feature_list)
    # bestMatrix = np.ndarray.tolist(bestMatrix)
    # bestMatrix.insert(0,feature_list)
    print np.array(feature_list)
    print bestMatrix
    format_str = 'Gen' + ' ' * 12 + '%%-8s  %%-8s  %%-8%s %%-10%s   %%-10%s   %%-10%s'
    print((format_str % ('s', 's', 's', 's')) % (
        'Max', 'Min', 'Avg', 'Best-Fscore', 'Best-Hamdist', 'Best-Accuracy'))


def checkAncients_callback(gp_engine):  # 只针对Gen 0
    if gp_engine.getCurrentGeneration() != 0:
        return

    delog.decache("check first Gen...")
    begin = 0
    end = gol.get_val("populationSize")
    population = gp_engine.getPopulation()
    for i in xrange(begin, end):
        genome = population[i]
        max_depth = genome.getParam("max_depth", None)

        # illegal?
        Illegal = False
        ecocMatrix, feature_list = TMConvertor.getMatrixDirectly_and_feature(genome)

        ###row###
        if LCheckers.sameRows(ecocMatrix):
            Illegal = True
        elif LCheckers.zeroRow(ecocMatrix):
            Illegal = True
        ###column###
        if LCheckers.tooLittleColumn(ecocMatrix):
            Illegal = True
        elif LCheckers.tooMuchColumn(ecocMatrix):
            Illegal = True

        if max_depth is None:
            Util.raiseException("You must specify the max_depth genome parameter !", ValueError)
        if max_depth < 0:
            Util.raiseException("The max_depth must be >= 1, if you want to use GTreeGPMutatorSubtree crossover !",
                                ValueError)

        while Illegal is True:
            new_genome = copy.deepcopy(genome)
            node = new_genome.getRandomNode()
            assert node is not None
            depth = new_genome.getNodeDepth(node)
            node_parent = node.getParent()
            root_subtree = GTree.buildGTreeGPGrow(gp_engine, 0, max_depth - depth)
            if node_parent is None:
                new_genome.setRoot(root_subtree)
            else:
                root_subtree.setParent(node_parent)
                node_parent.replaceChild(node, root_subtree)
            new_genome.processNodes()

            # illegal?
            Illegal = False
            ecocMatrix, feature_list = TMConvertor.getMatrixDirectly_and_feature(new_genome)

            ###row###
            if LCheckers.sameRows(ecocMatrix):
                Illegal = True
                continue
            elif LCheckers.zeroRow(ecocMatrix):
                Illegal = True
                continue
            ###column###
            if LCheckers.tooLittleColumn(ecocMatrix):
                Illegal = True
            elif LCheckers.tooMuchColumn(ecocMatrix):
                Illegal = True

            if Illegal == False:
                genome.setRoot(new_genome.getRoot())
                genome.processNodes()
    # Update the scores of population
    delog.deprint_string("over.")
    population.evaluate()
    population.sort()


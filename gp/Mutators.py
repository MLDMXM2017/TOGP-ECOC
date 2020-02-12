# -*- coding: utf-8 -*-
"""
:mod:`Mutators` -- mutation methods module
=====================================================================

In this module we have the genetic operators of mutation for each chromosome representation.

"""

import random, copy
from random import choice as rand_choice
from random import randint as rand_randint, gauss as rand_gauss, uniform as rand_uniform

import Consts
import GTree
import Util

from utils import globalVars as gol
from ecoc_core import TreeMatrixConvertor as TMConverter
from ecoc_core import LegalityCheckers as LC


###################
##     Tree gp   ##
###################

def GTreeGPMutatorOperation(genome, **args):
    """ The mutator of GTreeGP, Operation Mutator

    .. versionadded:: 0.6
       The *GTreeGPMutatorOperation* function
    """

    if args["pmut"] <= 0.0: return 0
    elements = len(genome)
    mutations = args["pmut"] * elements
    ga_engine = args["ga_engine"]

    gp_terminals = ga_engine.getParam("gp_terminals")
    assert gp_terminals is not None

    gp_function_set = ga_engine.getParam("gp_function_set")
    assert gp_function_set is not None

    if mutations < 1.0:
        mutations = 0
        for i in xrange(len(genome)):
            if Util.randomFlipCoin(args["pmut"]):
                mutations += 1
                rand_node = genome.getRandomNode()
                assert rand_node is not None
                if rand_node.getType() == Consts.nodeType["TERMINAL"]:
                    term_operator = rand_choice(gp_terminals)
                else:
                    op_len = gp_function_set[rand_node.getData()]
                    fun_candidates = []
                    for o, l in gp_function_set.items():
                        if l == op_len:
                            fun_candidates.append(o)

                    if len(fun_candidates) <= 0:
                        continue

                    term_operator = rand_choice(fun_candidates)
                rand_node.setData(term_operator)
    else:
        for it in xrange(int(round(mutations))):
            rand_node = genome.getRandomNode()
            assert rand_node is not None
            if rand_node.getType() == Consts.nodeType["TERMINAL"]:
                term_operator = rand_choice(gp_terminals)
            else:
                op_len = gp_function_set[rand_node.getData()]
                fun_candidates = []
                for o, l in gp_function_set.items():
                    if l == op_len:
                        fun_candidates.append(o)

                if len(fun_candidates) <= 0:
                    continue

                term_operator = rand_choice(fun_candidates)
            rand_node.setData(term_operator)

    return int(mutations)


"""   -by LKS  """


def DIYGTreeGPMutatorSubtree(genome, **args):
    """ The mutator of GTreeGP, Subtree Mutator

    This mutator will recreate random subtree of the tree using the grow algorithm.

    .. versionadded:: 0.6
       The *GTreeGPMutatorSubtree* function
    """
    classes = gol.get_val("classes")

    ind = genome
    Illegal = True
    while Illegal == True:

        # mutator
        if args["pmut"] <= 0.0: return 0
        ga_engine = args["ga_engine"]
        max_depth = genome.getParam("max_depth", None)
        mutations = 0

        if max_depth is None:
            Util.raiseException("You must specify the max_depth genome parameter !", ValueError)
        if max_depth < 0:
            Util.raiseException("The max_depth must be >= 1, if you want to use GTreeGPMutatorSubtree crossover !",
                                ValueError)

        node = genome.getRandomNode()
        assert node is not None
        if Util.randomFlipCoin(args["pmut"]):
            depth = genome.getNodeDepth(node)
            mutations += 1
            root_subtree = GTree.buildGTreeGPGrow(ga_engine, 0, max_depth - depth)
            node_parent = node.getParent()
            if node_parent is None:
                genome.setRoot(root_subtree)
            else:
                root_subtree.setParent(node_parent)
                node_parent.replaceChild(node, root_subtree)
            genome.processNodes()

        # illegal?
        Illegal = False
        ecocMatrix, feature_list = TMConverter.getMatrixDirectly_and_feature(ind)

        ###row###
        # 1.Two rows having the same numbers
        if LC.sameRows(ecocMatrix):
            Illegal = True
            continue
        # 2.There being a row with all 0
        elif LC.zeroRow(ecocMatrix):
            Illegal = True
            continue

        if LC.tooLittleColumn(ecocMatrix):
            Illegal = True
    return int(mutations)


"""   -by WHR  """


def selfDefined_GTreeGPMutatorSubtree(genome, **args):
    """
       The self defined mutator of GTreeGP, Subtree Mutator
       This mutator will recreate random subtree of the tree using the grow algorithm.
    """

    ind = genome
    Illegal = True
    classes = gol.get_val("classes")

    if args["pmut"] <= 0.0: return 0
    ga_engine = args["ga_engine"]
    max_depth = genome.getParam("max_depth", None)
    mutations = 0

    if max_depth is None:
        Util.raiseException("You must specify the max_depth genome parameter !", ValueError)
    if max_depth < 0:
        Util.raiseException("The max_depth must be >= 1, if you want to use GTreeGPMutatorSubtree crossover !",
                            ValueError)

    node = genome.getRandomNode()
    assert node is not None
    if Util.randomFlipCoin(args["pmut"]):
        depth = genome.getNodeDepth(node)
        node_parent = node.getParent()

        while Illegal is True:
            mutations += 1
            root_subtree = GTree.buildGTreeGPGrow(ga_engine, 0, max_depth - depth)
            if node_parent is None:
                genome.setRoot(root_subtree)
            else:
                root_subtree.setParent(node_parent)
                node_parent.replaceChild(node, root_subtree)
                node = root_subtree
            genome.processNodes()

            # illegal ? 其实 1 2 3可能不会出现
            Illegal = False
            ecocMatrix, feature_list = TMConverter.getMatrixDirectly_and_feature(genome)

            # 1.Two rows having the same numbers
            if LC.sameRows(ecocMatrix):
                Illegal = True
                continue
            # 2.There being a row with all 0
            elif LC.zeroRow(ecocMatrix):
                Illegal = True
                continue
            # 3.The number of column is too little
            elif LC.tooLittleColumn(ecocMatrix):
                Illegal = True

    return int(mutations)


"""   -by WHR  2018/03/05   """


def newDefined_GTreeGPMutatorSubtree(genome, **args):
    """
       The self defined mutator of GTreeGP, Subtree Mutator
       This mutator will recreate random subtree of the tree using the grow algorithm.
    """
    ga_engine = args["ga_engine"]
    max_depth = genome.getParam("max_depth", None)
    mutations = 0

    if max_depth is None:
        Util.raiseException("You must specify the max_depth genome parameter !", ValueError)
    if max_depth < 0:
        Util.raiseException("The max_depth must be >= 1, if you want to use GTreeGPMutatorSubtree crossover !",
                            ValueError)

    if Util.randomFlipCoin(args["pmut"]):
        Illegal = True
        while Illegal is True:
            new_genome = copy.deepcopy(genome)
            node = new_genome.getRandomNode()
            assert node is not None
            depth = new_genome.getNodeDepth(node)
            node_parent = node.getParent()
            mutations += 1
            root_subtree = GTree.buildGTreeGPGrow(ga_engine, 0, max_depth - depth)
            if node_parent is None:
                new_genome.setRoot(root_subtree)
            else:
                root_subtree.setParent(node_parent)
                node_parent.replaceChild(node, root_subtree)
            new_genome.processNodes()

            Illegal = False
            ecocMatrix, feature_list = TMConverter.getMatrixDirectly_and_feature(new_genome)

            # 1.Two rows having the same numbers
            if LC.sameRows(ecocMatrix):
                Illegal = True
                continue
            # 2.There being a row with all 0
            elif LC.zeroRow(ecocMatrix):
                Illegal = True
                continue
            # 3.The number of column is too little
            elif LC.tooLittleColumn(ecocMatrix):
                Illegal = True
        genome.setRoot(new_genome.getRoot())
        genome.processNodes()

    return int(mutations)

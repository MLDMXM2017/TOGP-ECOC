# -*- coding: utf-8 -*-
"""

:mod:`Crossovers` -- crossover methods module
=====================================================================

In this module we have the genetic operators of crossover (or recombination) for each chromosome representation.

"""

import math
from random import randint as rand_randint, choice as rand_choice
from random import random as rand_random

import Consts
from gp import Util


#############################################################################
#                  GTreeGP Crossovers  ######################################
#############################################################################

def GTreeGPCrossoverSinglePoint(genome, **args):
    """ The crossover of the GTreeGP, Single Point for Genetic Programming

    ..note:: This crossover method creates offspring with restriction of the
            *max_depth* parameter.

    Accepts the *max_attempt* parameter, *max_depth* (required).
    """
    # print "CrossoverAAAAAAAAAAA"
    sister = None
    brother = None

    gMom = args["mom"].clone()
    gDad = args["dad"].clone()

    gMom.resetStats()
    gDad.resetStats()

    max_depth = gMom.getParam("max_depth", None)
    max_attempt = gMom.getParam("max_attempt", 15)

    if max_depth is None:
        Util.raiseException("You must specify the max_depth genome parameter !", ValueError)

    if max_depth < 0:
        Util.raiseException(
            "The max_depth must be >= 1, if you want to use GTreeCrossoverSinglePointStrict crossover !", ValueError)

    momRandom = None
    dadRandom = None

    for i in xrange(max_attempt):

        dadRandom = gDad.getRandomNode()

        if dadRandom.getType() == Consts.nodeType["TERMINAL"]:
            momRandom = gMom.getRandomNode(1)
        elif dadRandom.getType() == Consts.nodeType["NONTERMINAL"]:
            momRandom = gMom.getRandomNode(2)

        mD = gMom.getNodeDepth(momRandom)
        dD = gDad.getNodeDepth(dadRandom)

        # Two nodes are root
        if mD == 0 and dD == 0:
            continue

        mH = gMom.getNodeHeight(momRandom)
        if dD + mH > max_depth:
            continue

        dH = gDad.getNodeHeight(dadRandom)
        if mD + dH > max_depth:
            continue

        break

    if i == (max_attempt - 1):
        assert gMom.getHeight() <= max_depth
        return gMom, gDad
    else:
        nodeMom, nodeDad = momRandom, dadRandom

    nodeMom_parent = nodeMom.getParent()
    nodeDad_parent = nodeDad.getParent()

    # Sister
    if args["count"] >= 1:
        sister = gMom
        nodeDad.setParent(nodeMom_parent)

        if nodeMom_parent is None:
            sister.setRoot(nodeDad)
        else:
            nodeMom_parent.replaceChild(nodeMom, nodeDad)
        sister.processNodes()
        assert sister.getHeight() <= max_depth

    # Brother
    if args["count"] == 2:
        brother = gDad
        nodeMom.setParent(nodeDad_parent)

        if nodeDad_parent is None:
            brother.setRoot(nodeMom)
        else:
            nodeDad_parent.replaceChild(nodeDad, nodeMom)
        brother.processNodes()
        assert brother.getHeight() <= max_depth

    return sister, brother


def NewGTreeGPCrossoverSinglePoint(genome, **args):
    sister = None
    brother = None

    gMom = args["mom"].clone()
    gDad = args["dad"].clone()

    gMom.resetStats()
    gDad.resetStats()

    max_depth = gMom.getParam("max_depth", None)
    max_attempt = gMom.getParam("max_attempt", 15)

    if max_depth is None:
        Util.raiseException("You must specify the max_depth genome parameter !", ValueError)

    if max_depth < 0:
        Util.raiseException("The max_depth must be >= 1, if you want to use GTreeCrossoverSinglePointStrict crossover !"
                            , ValueError)

    momRandom = None
    dadRandom = None

    for i in xrange(max_attempt):

        dadRandom = gDad.getRandomNode()

        if dadRandom.getType() == Consts.nodeType["TERMINAL"]:
            momRandom = gMom.getRandomNode(1)
        elif dadRandom.getType() == Consts.nodeType["NONTERMINAL"]:
            momRandom = gMom.getRandomNode(2)

        mD = gMom.getNodeDepth(momRandom)
        dD = gDad.getNodeDepth(dadRandom)

        # Two nodes are root
        if mD == 0 and dD == 0: continue

        mH = gMom.getNodeHeight(momRandom)
        if dD + mH > max_depth: continue

        dH = gDad.getNodeHeight(dadRandom)
        if mD + dH > max_depth: continue

        break

    if i == (max_attempt - 1):
        assert gMom.getHeight() <= max_depth
        return gMom, gDad
    else:
        nodeMom, nodeDad = momRandom, dadRandom

    nodeMom_parent = nodeMom.getParent()
    nodeDad_parent = nodeDad.getParent()

    # Sister
    if args["count"] >= 1:
        sister = gMom
        nodeDad.setParent(nodeMom_parent)

        if nodeMom_parent is None:
            sister.setRoot(nodeDad)
        else:
            nodeMom_parent.replaceChild(nodeMom, nodeDad)
        sister.processNodes()
        assert sister.getHeight() <= max_depth

    # Brother
    if args["count"] == 2:
        brother = gDad
        nodeMom.setParent(nodeDad_parent)

        if nodeDad_parent is None:
            brother.setRoot(nodeMom)
        else:
            nodeDad_parent.replaceChild(nodeDad, nodeMom)
        brother.processNodes()
        assert brother.getHeight() <= max_depth

    return sister, brother

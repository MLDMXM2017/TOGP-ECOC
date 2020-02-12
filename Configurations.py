# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 10:27
# @Author: Hanrui Wang
# @Target: Basic Configuration
import sys

version = "4.0"

dataName = "iris"

aimFolder = "ex1"

generations = 50

populationSize = 20

feature_number = 75  # the num of feature to be selected is 75 when feature space is larger than 75

n_jobs = 1

freq_stats = 1

n_neighbors = 3

crossoverRate = 1

mutationRate = 0.25

growMethod = "ramped"

if 'linux' in sys.platform:
    root_path = "/root/ECOC_Ternary/"

else:
    root_path = "./"


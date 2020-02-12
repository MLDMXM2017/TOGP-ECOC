# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 15:24
# @Author: Hanrui Wang


import numpy as np
from utils import globalVars as gol
from OutputCodeClassifier import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# (max_depth=None, min_samples_split=1,random_state=0)
# from sklearn.ensemble import RandomForestClassifier
# (n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
# from sklearn.ensemble import ExtraTreesClassifier
# (n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import BaggingClassifier


class ConnectClassifier:
    def __init__(self, features_used_list, codeMatrix):
        self.features_used_list = features_used_list
        self.code_Matrix = np.array(codeMatrix)
        self.features_used_list = features_used_list
        # self.estimator =  KNeighborsClassifier(n_neighbors=gol.get_val("n_neighbors"))
        # self.estimator =  GaussianNB()  
        # self.estimator = BernoulliNB()  
        # self.estimator =  LinearSVC(random_state=0) 
        self.estimator = DecisionTreeClassifier(random_state=0)
        # self.estimator = BaggingClassifier(random_state=0)
        self.oc = OutputCodeClassifier(self.estimator, self.code_Matrix, self.features_used_list, random_state=0)

    def TrainAndTest(self):
        self.oc.fit(self.features_used_list, self.code_Matrix)
        score, accuracy, text, new_ecocMatrix, new_features_used_list = self.oc.predict(self.features_used_list,
                                                                                        self.code_Matrix, 0)
        self.code_Matrix = new_ecocMatrix
        self.features_used_list = new_features_used_list
        return score, accuracy, text, new_ecocMatrix, new_features_used_list

    def TrainAndTestwithoutLocalImpr(self):
        self.oc.fit(self.features_used_list, self.code_Matrix)
        score, accuracy, pred = self.oc.predict(self.features_used_list, self.code_Matrix, 2)
        return score, accuracy

    def FinalTrainAndTest(self):
        self.oc.fit(self.features_used_list, self.code_Matrix)
        final_score, final_accuracy = self.oc.predictFinal(self.code_Matrix)
        return final_score, final_accuracy

# -*- coding: utf-8 -*-
# @Time  : 2018/3/4 15:38
# @Author: Hanrui Wang


from __future__ import division

import os

from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.metrics import confusion_matrix
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.pairwise import check_pairwise_arrays

import numpy as np
import warnings, sys
import copy, random
import LegalityCheckers as LC
from utils import delog
from utils import globalVars as gol
from utils.dirtools import check_folder
from utils.storage import Storager
from ecoc_core import DataComplexity

__all__ = ["OutputCodeClassifier"]


def get_distances(X, Y=None, Y_norm_squared=None, squared=False,
                  X_norm_squared=None):
    return corrected_euclidean_distances(X, Y)


def corrected_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                                  X_norm_squared=None):
    """
    X : predict values
    Y : code_book
    """
    X, Y = check_pairwise_arrays(X, Y)
    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    for i in xrange(X.shape[0]):  # for each sample
        for j in xrange(Y.shape[0]):  # calculate the dist between the sample and every base classifier
            row_x = np.copy(X[i])
            row_y = np.copy(Y[j])
            row_x[row_y == 0] = 0
            distances[i][j] += np.sum(row_x * row_x)
            distances[i][j] += np.sum(row_y * row_y)
            zero_num = row_y[row_y == 0].shape[0]
            if zero_num == 0:
                distances[i][j] /= row_x.shape[0]
            else:
                distances[i][j] /= (row_x.shape[0] - zero_num)
    np.maximum(distances, 0, out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    return distances if squared else np.sqrt(distances, out=distances)


def _fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y)
    return estimator


def corrected_fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    X, y = X[y != 0], y[y != 0]
    y[y == -1] = 0
    return _fit_binary(estimator, X, y)


def corrected_predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    score = _predict_binary(estimator, X)
    score *= 2
    score -= 1
    return score


def _partial_fit_binary(estimator, X, y):
    """Partially fit a single binary estimator."""
    estimator.partial_fit(X, y, np.array((0, 1)))
    return estimator


def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X)[:, 1]

    return score


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


def _sigmoid_normalize(X):  # sigmoid
    X = (X + 1) / 2
    return 1 / (1 + np.exp(-X)) * 2 - 1


class _ConstantPredictor(BaseEstimator):
    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')
        return np.repeat(self.y_, X.shape[0])

    def decision_function(self, X):
        check_is_fitted(self, 'y_')
        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')
        return np.repeat([np.hstack([1 - self.y_, self.y_])],
                         X.shape[0], axis=0)


class OutputCodeClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, estimator, ecocMatrix, features_used_list, random_state=None):

        self.Train_X = gol.get_val("Train_X")
        self.Train_Y = gol.get_val("Train_Y")
        self.Valid_X = gol.get_val("Valid_X")
        self.Valid_Y = gol.get_val("Valid_Y")
        self.Test_X = gol.get_val("Test_X")
        self.Test_Y = gol.get_val("Test_Y")

        n_jobs = gol.get_val("n_jobs")
        self.n_jobs = n_jobs
        self.estimator = estimator
        self.random_state = random_state
        self.classes_ = np.array(gol.get_val("classes"))
        self.sel_features = gol.get_val("sel_features")
        self.featuresNames = gol.get_val("feature_method")

        self.ecocMatrix = ecocMatrix
        self.features_used_list = features_used_list
        self.new_ecocMatrix = []
        self.new_features_used_list = []
        self.loglog = []
        self.fscore = 0
        self.accuracy = 0
        # save and get cache
        self.storager = Storager(gol.get_val("root_path"), gol.get_val("dataName"), self.estimator)

    def fit(self, features_used_list, ecocMatrix):
        """Fit underlying estimators.
        Parameters
        ----------
        Train_X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Train_Y : numpy array of shape [n_samples]
            Multi-class targets.
        Returns
        -------
        self
        """
        # prepare
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function' 
        else:
            self.estimator_type = 'predict_proba'  # output [0,1]
        feature_method_index = gol.get_val("feature_method_index")

        # class: index
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))

        ecocMatrix_extend = np.array([ecocMatrix[classes_index[self.Train_Y[i]]] for i in range(self.Train_X.shape[0])],
                                     dtype=np.int)

        # try to restore estimators from cache
        self.estimators_ = list()
        for i in range(ecocMatrix.shape[1]):
            _column = ecocMatrix[:, i]
            _features = feature_method_index[features_used_list[i]]
            self.storager.setfeaturecode(self.sel_features[_features], _column)
            est = self.storager.load_estimator_train()

            if est is None:
                # need training
                est = corrected_fit_binary(self.estimator, self.Train_X[:, self.sel_features[_features]],
                                           ecocMatrix_extend[:, i])
                self.storager.save_estimator_train(est)

            self.estimators_.append(est)
        return self

    def fit_one(self, X, y, feature, sel_features, code_book):
        # attention!!!!!!!!!!!!!!!!!!!!!
        # recheck if use this function
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'  
        else:
            self.estimator_type = 'predict_proba'  # output [0,1]

        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        Y = np.array([self.ecocMatrix[classes_index[y[i]]]
                      for i in range(X.shape[0])], dtype=np.int)

        feature_method_index = gol.get_val("feature_method_index")
        new_estimator = corrected_fit_binary(self.estimator, X[:, sel_features[feature_method_index[feature]]], Y[:, 0])

        self.estimators_.insert(len(self.estimators_), new_estimator)

        return self

    def predict(self, features_used_list, ecocMatrix, mark):
        """Predict multi-class targets using underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        # self.data_X = np.vstack((self.Train_X, self.Valid_X))
        # self.data_Y = np.hstack((self.Train_Y, self.Valid_Y))
        self.data_X = self.Valid_X
        self.data_Y = self.Valid_Y

        # prepare
        check_is_fitted(self, 'estimators_')
        feature_method_index = gol.get_val("feature_method_index")
        # try restore output from cache
        Y = []
        for i in xrange(len(self.estimators_)):
            _column = ecocMatrix[:, i]
            _features = feature_method_index[features_used_list[i]]
            self.storager.setfeaturecode(self.sel_features[_features], _column)
            pre = self.storager.load_prediction_valid()

            if pre is None:
                # need predicting
                pre = corrected_predict_binary(self.estimators_[i], self.data_X[:, self.sel_features[_features]])
                self.storager.save_prediction_valid(pre)
            Y.append(pre)

        Y = np.array(Y).T

        if self.estimator_type == 'decision_function':
            Y = _sigmoid_normalize(Y)

        pred = get_distances(Y, ecocMatrix).argmin(axis=1)

        self.fscore, self.accuracy = self.calculateFScore(self.classes_[pred], self.data_Y)

        # first use
        if mark is 0:
            hstr = "F-Score: " + str(self.fscore) + "\t\tAccuracy: " + str(self.accuracy)
            self.loglog.insert(len(self.loglog), self.features_used_list)
            self.loglog.insert(len(self.loglog), self.ecocMatrix)
            self.loglog.insert(len(self.loglog), hstr)
            new_fscore, new_accuracy = self.localImprovent_Ternary30(Y)
            self.loglog.insert(len(self.loglog), "-" * 100)
            return new_fscore, new_accuracy, self.loglog, self.new_ecocMatrix, self.new_features_used_list
            # return self.fscore, self.accuracy,  self.loglog, self.ecocMatrix, self.features_used_list
        if mark is 2:
            return self.fscore, self.accuracy, pred
        else:
            return self.fscore, self.accuracy

    def predictFinal(self, ecocMatrix):
        """Predict multi-class targets using underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        check_is_fitted(self, 'estimators_')
        feature_method_index = gol.get_val("feature_method_index")

        ########
        # TEST #
        ########
        # retraining because different training set, try to restore estimators from cache.
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        final_train_x = np.vstack((self.Train_X, self.Valid_X))
        final_train_y = np.hstack((self.Train_Y, self.Valid_Y))
        self.estimators_ = list()
        for i in range(ecocMatrix.shape[1]):
            _column = ecocMatrix[:, i]
            _features = feature_method_index[self.features_used_list[i]]
            self.storager.setfeaturecode(self.sel_features[_features], _column)
            est = self.storager.load_estimator_test()

            if est is None:
                # need training
                extend_column = np.array([_column[classes_index[final_train_y[i]]]
                                          for i in xrange(final_train_x.shape[0])], dtype=np.int)
                est = corrected_fit_binary(self.estimator, final_train_x[:, self.sel_features[_features]],
                                           extend_column)
                self.storager.save_estimator_test(est)

            self.estimators_.append(est)

        # predicting because different training set, try to restore output from cache.
        output_y = []
        for i in xrange(len(self.estimators_)):
            _column = ecocMatrix[:, i]
            _features = feature_method_index[self.features_used_list[i]]
            self.storager.setfeaturecode(self.sel_features[_features], _column)
            pre = self.storager.load_prediction_test()
            if pre is None:
                pre = corrected_predict_binary(self.estimators_[i], self.Test_X[:, self.sel_features[_features]])
                self.storager.save_prediction_test(pre)
            output_y.append(pre)
        output_y = np.array(output_y).T
        if self.estimator_type == 'decision_function':
            output_y = _sigmoid_normalize(output_y)  # 采用sigmoid函数是因为 decision_function 产生的结果的范围是负无穷到正无穷

        # get score
        pred = get_distances(output_y, ecocMatrix).argmin(axis=1)
        score, accuracy = self.calculateFScore(self.classes_[pred], self.Test_Y)

        return score, accuracy

    def calculateFScore(self, pred, validation_labels):
        accuracy = (np.float)(pred[pred == validation_labels].shape[0]) / (np.float)(pred.shape[0])
        scoreList = precision_recall_fscore_support(validation_labels, pred)[2]

        # record every class's fscore when final test
        if gol.get_val("final") == 1:
            filedir = gol.get_val("root_path")
            filedir = os.path.join(filedir, 'Results/' + gol.get_val("version"))
            filedir = os.path.join(filedir, gol.get_val("dataName") + "-v" + gol.get_val("version"))
            check_folder(filedir)
            file = open(str(filedir) + "/fscoreEachclass.csv", 'a')
            for i in xrange(len(scoreList)):
                file.write(str(scoreList[i]) + ',')
            file.write('\n')

        score = np.average(scoreList)

        return score, accuracy

    # version 1.0
    def localImprovent_Ternary(self, Y):

        lenCol = self.ecocMatrix.shape[1]
        classes = gol.get_val("classes")
        classDict = dict((j, i) for i, j in enumerate(classes))
        labelCount = [list(self.Valid_Y).count(i) for i in classes]  # 每种类别的个数
        diversity_matrix = [[0 for col in range(lenCol)] for row in range(lenCol)]  # 多样性矩阵

        # calculate True or False matrix
        # each row corresponds to Valid_Y, each column represents the judgement of classifier whether correct.
        Y_Bool = []
        for i in xrange(len(Y)):
            temp = []
            for j in xrange(len(Y[i])):
                labelNum = classDict[self.Valid_Y[i]]
                temp.append(int(self.ecocMatrix[labelNum][j]) is int(Y[i][j]))
            Y_Bool.append(temp)

        # calculate diversity matrix
        for i in xrange(lenCol):
            labelSet = list([0 for p in xrange(len(classes))])
            for j in xrange(i):
                for k in xrange(len(Y_Bool)):
                    if Y_Bool[k][i] is not Y_Bool[k][j]:
                        labelNum = classDict[self.Valid_Y[k]]
                        labelSet[labelNum] += 1
                for l in xrange(len(labelSet)):
                    labelSet[l] /= labelCount[l]
                diversity_matrix[i][j] = np.sum(labelSet)
                diversity_matrix[j][i] = diversity_matrix[i][j]
        # diversity_matrix = np.array(diversity_matrix)
        self.loglog.insert(len(self.loglog), "\nDiversity_Matrix: ")
        self.loglog.insert(len(self.loglog), np.array(diversity_matrix))

        # calculate the score(variance) of each classifier
        finalSum = [0] * lenCol
        for i in xrange(lenCol):
            d = [0] * len(classes)
            for k in xrange(len(self.Valid_Y)):
                labelNum = classDict[self.Valid_Y[k]]
                d[labelNum] += np.square(Y[k][i] - self.ecocMatrix[labelNum][i])
            for l in xrange(len(classes)):
                finalSum[i] += float(d[l] / labelCount[l] / labelCount[l])
        self.loglog.insert(len(self.loglog), "\nScore(Variance) of each classifier: ")
        self.loglog.insert(len(self.loglog), np.array(finalSum))

        # select classifiers which have the top n-1 score
        minIndex = finalSum.index(min(finalSum))
        for i in xrange(len(classes) - 1):
            finalSum[minIndex] = sys.maxint
            self.new_ecocMatrix.append(list(self.ecocMatrix[:, minIndex]))
            self.new_features_used_list.append(self.features_used_list[minIndex])
            minIndex = finalSum.index(min(finalSum))
        self.fit(self.new_features_used_list, np.array(self.new_ecocMatrix).transpose())
        score, accuracy = self.predict(self.new_features_used_list, np.array(self.new_ecocMatrix).transpose(), 1)

        self.loglog.insert(len(self.loglog), "\n\nBriefly ECOC_Ternary Matrix: ")
        self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
        self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix).transpose())
        self.loglog.insert(len(self.loglog), "F-Score: " + str(score) + "\t\tAccuracy: " + str(accuracy) + "\n")

        # Add other columns
        def get_maxDiversity():
            maxs = [max(i) for i in diversity_matrix]
            if max(maxs) is 0: return None, None
            maxIndexCol = maxs.index(max(maxs))
            maxIndexRow = diversity_matrix[:][maxIndexCol].index(max(maxs))
            hstr = "Two columns which have the top diversity:\n" \
                   + str(maxIndexCol) + str(self.ecocMatrix[:, maxIndexCol]) + "\n" \
                   + str(maxIndexRow) + str(self.ecocMatrix[:, maxIndexRow])
            self.loglog.insert(len(self.loglog), hstr)
            diversity_matrix[maxIndexCol][maxIndexRow] = 0
            diversity_matrix[maxIndexRow][maxIndexCol] = 0
            return maxIndexCol, maxIndexRow

        maxIndexCol, maxIndexRow = get_maxDiversity()
        classifier_1 = self.ecocMatrix[:, maxIndexCol]
        classifier_2 = self.ecocMatrix[:, maxIndexRow]
        while True:
            if (maxIndexCol is None) & (maxIndexRow is None): break
            if (classifier_1 in np.array(self.new_ecocMatrix)) & (classifier_2 in np.array(self.new_ecocMatrix)):
                maxIndexCol, maxIndexRow = get_maxDiversity()
                classifier_1 = self.ecocMatrix[:, maxIndexCol]
                classifier_2 = self.ecocMatrix[:, maxIndexRow]
                continue
            elif (classifier_1 not in np.array(self.new_ecocMatrix)) & (
                    classifier_2 not in np.array(self.new_ecocMatrix)):
                maxIndexCol, maxIndexRow = get_maxDiversity()
                classifier_1 = self.ecocMatrix[:, maxIndexCol]
                classifier_2 = self.ecocMatrix[:, maxIndexRow]
                continue
            elif classifier_1 in np.array(self.new_ecocMatrix):
                self.new_ecocMatrix.append(classifier_2)
                self.new_features_used_list.append(self.features_used_list[maxIndexRow])
                self.fit(np.array(self.new_features_used_list), np.array(self.new_ecocMatrix).transpose())
                new_score, new_acc = self.predict(self.new_features_used_list,
                                                  np.array(self.new_ecocMatrix).transpose(), 1)
                if new_acc > accuracy:
                    self.loglog.insert(len(self.loglog), "After adding column2: ")
                    self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
                    self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix).transpose())
                    self.loglog.insert(len(self.loglog),
                                       "F-Score: " + str(new_score) + "\t\tAccuracy: " + str(new_acc) + "\n")
                    classifier_1, classifier_2 = get_maxDiversity()
                    continue
                break
            else:
                self.new_ecocMatrix.append(classifier_1)
                self.new_features_used_list.append(self.features_used_list[maxIndexCol])
                self.fit(self.new_features_used_list, np.array(self.new_ecocMatrix).transpose())
                new_score, new_acc = self.predict(self.new_features_used_list,
                                                  np.array(self.new_ecocMatrix).transpose(), 1)
                if new_acc > accuracy:
                    self.loglog.insert(len(self.loglog), "After adding column1: ")
                    self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
                    self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix).transpose())
                    self.loglog.insert(len(self.loglog),
                                       "F-Score: " + str(new_score) + "\t\tAccuracy: " + str(new_acc) + "\n")
                    classifier_1, classifier_2 = get_maxDiversity()
                    continue
                break

        self.new_ecocMatrix = np.array(self.new_ecocMatrix).transpose()
        self.fit(self.new_features_used_list, self.new_ecocMatrix)
        new_fscore, new_accuracy = self.predict(self.new_features_used_list, self.new_ecocMatrix, 1)

        self.loglog.insert(len(self.loglog), "\n\nFinal ECOC coding matrix: ")
        self.loglog.insert(len(self.loglog), self.new_features_used_list)
        self.loglog.insert(len(self.loglog), self.new_ecocMatrix)
        self.loglog.insert(len(self.loglog), "F-Score: " + str(new_fscore) + "\t\tAccuracy: " + str(new_accuracy))

        return new_fscore, new_accuracy

    # version 2.0
    def localImprovent_Ternary20(self, Y):

        lenCol = self.ecocMatrix.shape[1]
        lenRow = self.ecocMatrix.shape[0]
        classes = gol.get_val("classes")
        classDict = dict((j, i) for i, j in enumerate(classes))
        labelCount = [list(self.Valid_Y).count(i) for i in classes]  # 每种类别的个数
        Y_Bool = []
        KappaList = []
        diversity_matrix = [0 for row in range(lenCol)]  # 多样性矩阵

        # calculate Y_Bool: True or False matrix
        # each row corresponds to Valid_Y, each column represents the judgement of classifier whether correct.
        for i in xrange(len(Y)):
            temp = []
            for j in xrange(len(Y[i])):
                labelNum = classDict[self.Valid_Y[i]]
                if Y[i][j] > 0:
                    temp.append(int(self.ecocMatrix[labelNum][j]) is 1)
                else:
                    temp.append(int(self.ecocMatrix[labelNum][j]) is -1)
            Y_Bool.append(temp)

        # calculate Kappa List, according to the formula2
        for i in xrange(lenCol):
            conMatrix = [[0 for row in range(2)] for row in range(2)]
            for k in xrange(len(Y_Bool)):
                labelNum = classDict[self.Valid_Y[k]]
                if int(self.ecocMatrix[labelNum][i]) is 1:
                    if Y_Bool[k][i] is True:
                        conMatrix[0][0] += 1
                    else:
                        conMatrix[0][1] += 1
                elif int(self.ecocMatrix[labelNum][i]) is -1:
                    if Y_Bool[k][i] is True:
                        conMatrix[1][1] += 1
                    else:
                        conMatrix[1][0] += 1
            up = conMatrix[0][0] - np.sum(conMatrix[0][:]) * np.sum(conMatrix[:][0])
            if int(up) is 0:
                leftKappa = 0
            else:
                leftKappa = (conMatrix[0][0] - np.sum(conMatrix[0][:]) * np.sum(conMatrix[:][0])) / (
                        np.sum(conMatrix[0][:]) - np.sum(conMatrix[0][:]) * np.sum(conMatrix[:][0]))
            up = conMatrix[1][1] - np.sum(conMatrix[1][:]) * np.sum(conMatrix[:][1])
            if int(up) is 0:
                rightKappa = 0
            else:
                rightKappa = (conMatrix[1][1] - np.sum(conMatrix[1][:]) * np.sum(conMatrix[:][1])) / (
                        np.sum(conMatrix[1][:]) - np.sum(conMatrix[1][:]) * np.sum(conMatrix[:][1]))
            KappaList.append(np.mean([leftKappa, rightKappa]))
        self.loglog.insert(len(self.loglog), "\nKappa List: ")
        self.loglog.insert(len(self.loglog), np.array(KappaList))

        # calculate the score(variance) of each classifier
        finalSum = [0] * lenCol
        for i in xrange(lenCol):
            d = [0] * len(classes)
            for k in xrange(len(self.Valid_Y)):
                labelNum = classDict[self.Valid_Y[k]]
                d[labelNum] += np.square(Y[k][i] - self.ecocMatrix[labelNum][i])
            for l in xrange(len(classes)):
                finalSum[i] += float(d[l] / labelCount[l] / labelCount[l])
        self.loglog.insert(len(self.loglog), "\nScore(Variance) of each classifier: ")
        self.loglog.insert(len(self.loglog), np.array(finalSum))
        self.loglog.insert(len(self.loglog), "\n\n")

        # Greedy ensemble purning algorithm
        minIndex = KappaList.index(min(KappaList))
        self.new_ecocMatrix = self.ecocMatrix
        self.new_features_used_list = self.features_used_list
        score, accuracy = self.score, self.accuracy
        while True:

            tempF = np.delete(self.new_features_used_list, minIndex, axis=0)
            tempE = np.delete(self.new_ecocMatrix, minIndex, axis=1)
            if len(self.new_features_used_list) < 2:
                self.loglog.insert(len(self.loglog), "Final:\n")
                self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
                self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix))
                break

            self.fit(tempF, tempE)
            new_score, new_acc = self.predict(tempF, tempE, 1)

            self.loglog.insert(len(self.loglog),
                               "Prepare to delete column" + str(minIndex) + ":" + str(self.new_ecocMatrix[:, minIndex]))
            self.loglog.insert(len(self.loglog), "F-Score: " + str(new_score) + "\t\tAccuracy: " + str(new_acc) + "\n")

            if new_acc < accuracy:
                self.loglog.insert(len(self.loglog), "Final:\n")
                self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
                self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix))
                break

            self.new_ecocMatrix = tempE
            self.new_features_used_list = tempF
            KappaList = list(np.delete(KappaList, minIndex, axis=0))
            minIndex = KappaList.index(min(KappaList))

        self.fit(self.new_features_used_list, np.array(self.new_ecocMatrix))
        new_fscore, new_accuracy = self.predict(self.new_features_used_list, np.array(self.new_ecocMatrix), 1)
        self.loglog.insert(len(self.loglog),
                           "F-Score: " + str(new_fscore) + "\t\tAccuracy: " + str(new_accuracy) + "\n")

        return new_fscore, new_accuracy

    # version 2.1
    def localImprovent_Ternary21(self, Y):

        lenCol = self.ecocMatrix.shape[1]
        lenRow = self.ecocMatrix.shape[0]
        classes = gol.get_val("classes")
        classDict = dict((j, i) for i, j in enumerate(classes))
        labelCount = [list(self.Valid_Y).count(i) for i in classes]  # 每种类别的个数
        Y_Bool = []
        diversity_matrix = [0 for row in range(lenCol)]  # 多样性矩阵

        # calculate Y_Bool: True or False matrix
        # each row corresponds to Valid_Y, each column represents the judgement of classifier whether correct.
        for i in xrange(len(Y)):
            temp = []
            for j in xrange(len(Y[i])):
                labelNum = classDict[self.Valid_Y[i]]
                if Y[i][j] > 0:
                    temp.append(int(self.ecocMatrix[labelNum][j]) is 1)
                else:
                    temp.append(int(self.ecocMatrix[labelNum][j]) is -1)
            Y_Bool.append(temp)

        # calculate the score(variance) of each classifier
        finalSum = [0] * lenCol
        for i in xrange(lenCol):
            d = [0] * len(classes)
            for k in xrange(len(self.Valid_Y)):
                labelNum = classDict[self.Valid_Y[k]]
                d[labelNum] += np.square(Y[k][i] - self.ecocMatrix[labelNum][i])
            for l in xrange(len(classes)):
                finalSum[i] += float(d[l] / labelCount[l] / labelCount[l])
        self.loglog.insert(len(self.loglog), "\nScore(Variance) of each classifier: ")
        self.loglog.insert(len(self.loglog), np.array(finalSum))
        self.loglog.insert(len(self.loglog), "\n\n")

        # calculate Kappa List, according to the formula1
        def calculate_KappaList(ecocMatrix, features_used_list):

            KappaList = []
            check_is_fitted(self, 'estimators_')
            feature_method_index = gol.get_val("feature_method_index")
            Y = np.array([corrected_predict_binary(self.estimators_[i], self.Valid_X[:, self.sel_features[
                                                                                            feature_method_index[
                                                                                                features_used_list[
                                                                                                    i]]]])
                          for i in xrange(len(self.estimators_))]).T
            if self.estimator_type == 'decision_function':
                Y = _sigmoid_normalize(Y)  # 采用sigmoid函数是因为 decision_function 产生的结果的范围是负无穷到正无穷
            pred = get_distances(Y, ecocMatrix).argmin(axis=1)
            for i in xrange(len(pred)):
                if pred[i] == classDict[self.Valid_Y[i]]:
                    pred[i] = True
                else:
                    pred[i] = False
            for i in xrange(ecocMatrix.shape[1]):
                conMatrix = [[0 for row in range(2)] for row in range(2)]
                for k in xrange(len(Y_Bool)):
                    if (Y_Bool[k][i] is True) & (bool(pred[i]) is True):
                        conMatrix[0][0] += 1
                    elif (Y_Bool[k][i] is True) & (bool(pred[i]) is False):
                        conMatrix[0][1] += 1
                    elif (Y_Bool[k][i] is False) & (bool(pred[i]) is True):
                        conMatrix[1][0] += 1
                    else:
                        conMatrix[1][1] += 1
                Pc = ((conMatrix[0][0] + conMatrix[0][1]) * (conMatrix[0][0] + conMatrix[1][0]) + (
                        conMatrix[1][0] + conMatrix[1][1]) * (conMatrix[0][1] + conMatrix[1][1])) / len(
                    Y_Bool) / len(Y_Bool)
                P0 = (conMatrix[0][0] + conMatrix[1][1]) / len(Y_Bool)
                # print conMatrix
                # print P0, Pc
                down = 1 - Pc
                if down == 0:
                    Kappa = 1
                else:
                    Kappa = (P0 - Pc) / (1 - Pc)
                KappaList.append(Kappa)
            # print KappaList
            self.loglog.insert(len(self.loglog), "Kappa List: ")
            self.loglog.insert(len(self.loglog), np.array(KappaList))
            return KappaList

        # Greedy ensemble purning algorithm
        self.new_ecocMatrix = self.ecocMatrix
        self.new_features_used_list = self.features_used_list
        self.pred = get_distances(Y, self.ecocMatrix).argmin(axis=1)
        score, accuracy = self.score, self.accuracy
        KappaList = calculate_KappaList(self.new_ecocMatrix, self.new_features_used_list)
        minIndex = KappaList.index(min(KappaList))
        while True:

            tempF = np.delete(self.new_features_used_list, minIndex, axis=0)
            tempE = np.delete(self.new_ecocMatrix, minIndex, axis=1)
            if len(self.new_features_used_list) < 2:
                self.loglog.insert(len(self.loglog), "\nFinal:\n")
                self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
                self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix))
                break

            self.fit(tempF, tempE)
            new_score, new_acc = self.predict(tempF, tempE, 1)

            self.loglog.insert(len(self.loglog),
                               "Prepare to delete column" + str(minIndex) + ":" + str(self.new_ecocMatrix[:, minIndex]))
            self.loglog.insert(len(self.loglog), "F-Score: " + str(new_score) + "\t\tAccuracy: " + str(new_acc) + "\n")

            if new_acc < accuracy:
                self.loglog.insert(len(self.loglog), "\nFinal:\n")
                self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
                self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix))
                break

            self.new_ecocMatrix = tempE
            self.new_features_used_list = tempF
            KappaList = calculate_KappaList(tempE, tempF)
            minIndex = KappaList.index(min(KappaList))

        self.fit(self.new_features_used_list, np.array(self.new_ecocMatrix))
        new_fscore, new_accuracy = self.predict(self.new_features_used_list, np.array(self.new_ecocMatrix), 1)
        self.loglog.insert(len(self.loglog),
                           "F-Score: " + str(new_fscore) + "\t\tAccuracy: " + str(new_accuracy) + "\n")

        return new_fscore, new_accuracy

    # version 3.0
    def localImprovent_Ternary30(self, Y):

        lenRow = self.ecocMatrix.shape[0]
        lenCol = self.ecocMatrix.shape[1]
        classes = gol.get_val("classes")
        classDict = dict((j, i) for i, j in enumerate(classes))
        labelCount = [list(self.data_Y).count(i) for i in classes]  # 每种类别的个数

        # calculate True or False matrix
        # each row corresponds to Valid_Y, each column represents the judgement of classifier whether correct.
        def calculate_YBool():
            Y_Bool = []
            for i in xrange(len(Y)):
                temp = []
                for j in xrange(len(Y[i])):
                    labelNum = classDict[self.data_Y[i]]
                    if Y[i][j] > 0:
                        temp.append(int(self.ecocMatrix[labelNum][j]) is 1)
                    else:
                        temp.append(int(self.ecocMatrix[labelNum][j]) is -1)
                Y_Bool.append(temp)
            return Y_Bool

        # calculate the accuracy of each classifier
        def calculate_classifierAcc(Y_Bool):
            classifierAcc = []
            Y_Bool = np.array(Y_Bool)
            for i in xrange(lenCol):
                acc = list(Y_Bool[:, i]).count(True) / Y_Bool.shape[0]
                classifierAcc.append(acc)
            self.loglog.insert(len(self.loglog), "\nAccuracy of each classifier: ")
            self.loglog.insert(len(self.loglog), np.array(classifierAcc))
            return classifierAcc

        # select classifiers which have the top n-1 accuracy
        def briefly_ECOC(classifierAcc):

            maxIndex = classifierAcc.index(max(classifierAcc))
            for i in xrange(len(classes) - 1):
                classifierAcc[maxIndex] = -1
                self.new_ecocMatrix.append(list(self.ecocMatrix[:, maxIndex]))
                self.new_features_used_list.append(self.features_used_list[maxIndex])
                maxIndex = classifierAcc.index(max(classifierAcc))
            self.fit(self.new_features_used_list, np.array(self.new_ecocMatrix).transpose())
            score, accuracy = self.predict(self.new_features_used_list, np.array(self.new_ecocMatrix).transpose(), 1)

            self.loglog.insert(len(self.loglog), "\n\nBriefly ECOC_Ternary Matrix: ")
            self.loglog.insert(len(self.loglog), np.array(self.new_features_used_list))
            self.loglog.insert(len(self.loglog), np.array(self.new_ecocMatrix).transpose())
            self.loglog.insert(len(self.loglog), "F-Score: " + str(score) + "\t\tAccuracy: " + str(accuracy) + "\n")

        # To select two classes which are confused most (if return -1,-1 ,then do nothing)
        def findMaxByConfusion():
            max = 0
            maxi = -1
            maxj = -1
            for i in xrange(self.conMatrix.shape[0]):
                for j in xrange(self.conMatrix.shape[1]):
                    if j > i:
                        if self.conMatrix[i][j] + self.conMatrix[j][i] > max:
                            max = self.conMatrix[i][j] + self.conMatrix[j][i]
                            maxi = i
                            maxj = j
            if maxi != -1 & maxj != -1:
                self.conMatrix[maxi][maxj] = 0
                self.conMatrix[maxj][maxi] = 0
            return maxi, maxj

        # calculate self-defined index
        def calculate_SDAcc(maxi, maxj, pred):
            SDAcc = [0] * len(classifierAcc)

            candidate = [i != -1 for i in classifierAcc]
            for i in xrange(len(Y)):
                if (classDict[self.data_Y[i]] == maxi) | (classDict[self.data_Y[i]] == maxj):
                    if pred[i] != classDict[self.data_Y[i]]:
                        for k in xrange(len(candidate)):
                            if candidate[k] & Y_Bool[i][k]:
                                NF = Y_Bool[i].count(True) / len(classifierAcc)
                                SDAcc[k] += NF
            self.loglog.insert(len(self.loglog), "\nSDAcc of each classifier: ")
            self.loglog.insert(len(self.loglog), np.array(SDAcc))
            return SDAcc

        # get max SDAcc
        def get_maxSDAcc():
            if max(SDAcc) is 0: return None
            maxIndex = SDAcc.index(max(SDAcc))
            hstr = "Prepare to add this column:" + str(maxIndex) + "\t" + str(self.ecocMatrix[:, maxIndex]) + "\n"
            self.loglog.insert(len(self.loglog), hstr)
            SDAcc[maxIndex] = 0
            classifierAcc[maxIndex] = -1
            return maxIndex

        Y_Bool = calculate_YBool()
        classifierAcc = calculate_classifierAcc(Y_Bool)
        briefly_ECOC(classifierAcc)
        # print np.array(self.new_features_used_list)
        # print np.array(self.new_ecocMatrix).transpose()
        # print self.conMatrix
        # print "#"*30
        self.fscore, self.accuracy, pred = self.predict(self.new_features_used_list,
                                                        np.array(self.new_ecocMatrix).transpose(), 2)
        self.conMatrix = confusion_matrix(self.data_Y, self.classes_[pred])
        self.loglog.insert(len(self.loglog), confusion_matrix(self.data_Y, self.classes_[pred]))
        while True:
            maxi, maxj = findMaxByConfusion()
            if maxi == -1 & maxj == -1: break
            self.loglog.insert(len(self.loglog), str(maxi) + "\t" + str(maxj))
            SDAcc = calculate_SDAcc(maxi, maxj, pred)

            # bool_array = np.zeros(len(self.Valid_Y), dtype=np.bool)  # generate a array being filled of 'False'
            # bool_array = bool_array | np.array(self.Valid_Y == classes[maxi]) | np.array(self.Valid_Y == classes[maxj])
            # part_Valid_X = self.Valid_X[bool_array]
            # part_Valid_Y = self.Valid_Y[bool_array]

            maxIndex = get_maxSDAcc()
            if maxIndex is None: continue
            self.new_features_used_list.append(self.features_used_list[maxIndex])
            self.new_ecocMatrix.append(self.ecocMatrix[:, maxIndex])

        self.new_features_used_list = np.array(self.new_features_used_list)
        self.new_ecocMatrix = np.array(self.new_ecocMatrix).transpose()
        self.loglog.insert(len(self.loglog), "\nFinal:\n")
        self.loglog.insert(len(self.loglog), self.new_features_used_list)
        self.loglog.insert(len(self.loglog), self.new_ecocMatrix)

        # print self.new_features_used_list
        # print self.new_ecocMatrix
        self.fit(self.new_features_used_list, self.new_ecocMatrix)
        new_fscore, new_accuracy = self.predict(self.new_features_used_list, self.new_ecocMatrix, 1)
        self.loglog.insert(len(self.loglog),
                           "F-Score: " + str(new_fscore) + "\t\tAccuracy: " + str(new_accuracy) + "\n")

        return new_fscore, new_accuracy

    def localImprovment(self, param_Y, classes, labels, sel_features):
        # caculate the effiency of each column
        finalSum = [0] * (self.ecocMatrix.shape[1])
        for i in xrange(self.ecocMatrix.shape[1]):
            for j in xrange(len(classes)):
                d = 0
                for k in xrange(len(labels)):
                    if classes[j] == labels[k]:
                        d += np.square(param_Y[k][i] - self.ecocMatrix[j][i])
                num = (list(labels)).count(classes[j])
                finalSum[i] += float(d / num / num)
                # finalSum[i] = finalSum[i]/len(classes)

        ########################################
        ##     hanrui's modifier 2017.07.19   ##
        ########################################
        dataComplexity = [0] * (self.ecocMatrix.shape[1])
        for i in xrange(self.ecocMatrix.shape[1]):
            column = [0] * (self.ecocMatrix.shape[0])
            for j in xrange(len(classes)):
                column[j] = self.ecocMatrix[j][i]
            dataComplexity[i] = self.calculateDataComplexity(column, self.feature_name[i])

        for i in xrange(self.ecocMatrix.shape[1]):
            if dataComplexity[i] != 0:
                finalSum[i] = float(finalSum[i] / dataComplexity[i])
            else:
                finalSum[i] = 0x7fffffff

        # delete one most terrible column
        while True:

            Y = copy.deepcopy(param_Y)
            ecocMatrix = copy.deepcopy(self.ecocMatrix)
            estimator = copy.deepcopy(self.estimators_)
            feature_temp = copy.deepcopy(self.feature_name)
            if len(finalSum):
                # delete the min efficiency column, that means the highest distence
                m = max(finalSum)
                minColumn = (list(finalSum)).index(m)

                deletes = [minColumn]
                ecocMatrix = np.delete(ecocMatrix, deletes, axis=1)
                Y = np.delete(Y, deletes, axis=1)
                del estimator[minColumn]
                # estimator = np.delete(estimator, deletes, axis=0)
                feature_temp = np.delete(feature_temp, deletes, axis=0)

                # judge if this delete would make the codeMatrix not valid
                if LC.sameRows(ecocMatrix):
                    finalSum = np.delete(finalSum, deletes, axis=0)
                    continue
                elif LC.zeroRow(ecocMatrix):
                    finalSum = np.delete(finalSum, deletes, axis=0)
                    continue
                elif LC.tooLittleColumn(ecocMatrix):
                    finalSum = np.delete(finalSum, deletes, axis=0)
                    continue
                else:
                    text = 'Prepare to delete this column: ' + str(minColumn) + '\t score = ' + str(
                        finalSum[minColumn] * dataComplexity[minColumn]) + '\t dataComplexity = ' + str(
                        dataComplexity[minColumn])
                    self.loglog.insert(len(self.loglog), text)
                    break

            # none aviliable column could be deleted
            else:
                # print "minColumn: None"
                break

        return Y, ecocMatrix, estimator, feature_temp

    def selectBestGroup(self, maxi, maxj):
        addColumn = np.zeros(len(self.classes))
        addColumn[maxi] = -1
        addColumn[maxj] = 1

        self.finalColumn = copy.deepcopy(addColumn)
        # the function hasExist
        matrix = self.ecocMatrix.transpose()
        _bool_var = (self.finalColumn == matrix).all(axis=1) | (self.finalColumn * -1 == matrix).all(axis=1)
        res = self.feature_name[_bool_var.tolist().index(True)] if _bool_var.any() else False
        if (res == False):
            self.finalFeature = random.choice(self.featuresNames)  # select feature randomly
        else:
            temp = copy.deepcopy(self.featuresNames)
            temp.remove(res)
            self.finalFeature = random.choice(temp)
        self.complexity = self.calculateDataComplexity(addColumn, self.finalFeature)
        # import datetime,sys
        # time1 = datetime.datetime.now()
        # for k in xrange(20):
        self.Cartesian_test(addColumn, maxi, maxj)
        # print (datetime.datetime.now() - time1)
        # time1 = datetime.datetime.now()
        # for k in xrange(20):
        #     self.Cartesian_test(addColumn)
        # print (datetime.datetime.now() - time1)
        # sys.exit(0)
        return self.finalColumn, self.finalFeature

    # Cartesian product find the better Group
    def Cartesian(self, column, maxi, maxj):
        '''
        # x, y and z are simple iterator
        # min_tuples are used to pruning
        '''
        baseList = [-1, 0, 1]
        listNum = len(column) - 2
        columnCopy = copy.deepcopy(column)

        import itertools
        spaces = itertools.product(baseList, repeat=listNum)
        for x in spaces:
            y = 0
            for z in xrange(len(column)):
                if (columnCopy[z] == 0):
                    column[z] = x[y]
                    y = y + 1

            # the function hasExist
            matrix = self.ecocMatrix.transpose()
            _bool_var = (column == matrix).all(axis=1) | (column * -1 == matrix).all(axis=1)
            res = self.feature_name[_bool_var.tolist().index(True)] if _bool_var.any() else False
            if (res == False):
                self.finalFeature = random.choice(self.featuresNames)
            else:
                temp = copy.deepcopy(self.featuresNames)
                temp.remove(res)
                self.finalFeature = random.choice(temp)

            newComplexity = self.calculateDataComplexity(column, self.finalFeature)
            if newComplexity > self.complexity:
                self.complexity = newComplexity
                self.finalColumn = copy.deepcopy(column)

    # Cartesian product find the better Group
    def Cartesian_test(self, column, maxi, maxj):
        '''
        # x, y and z are simple iterator
        # min_tuples are used to pruning
        '''
        baseList = [-1, 0, 1]
        listNum = len(column) - 2
        columnCopy = copy.deepcopy(column)

        # pruning
        pruning_index = set()
        confusion = self.conMatrix
        for i in xrange(len(column)):
            if i == maxi or i == maxj:    continue
            # can not tell difference between i and maxi,  as well as i and maxj
            if confusion[i][maxi] + confusion[maxi][i] > 0 and confusion[i][maxj] + confusion[maxj][i] > 0:
                pruning_index.add(i)
            elif confusion[i][maxi] + confusion[maxi][i] == 0 and confusion[i][maxj] + confusion[maxj][i] == 0:
                pruning_index.add(i)
        listNum -= len(pruning_index)

        import itertools
        spaces = itertools.product(baseList, repeat=listNum)
        for x in spaces:
            y = 0
            for z in xrange(len(column)):
                # pruning
                if z not in pruning_index and columnCopy[z] == 0:
                    column[z] = x[y]
                    y = y + 1

            # column = np.insert(x, (maxi, maxj - 1), (-1, 1))

            # the function hasExist
            matrix = self.ecocMatrix.transpose()
            _bool_var = (column == matrix).all(axis=1) | (column * -1 == matrix).all(axis=1)
            res = self.feature_name[_bool_var.tolist().index(True)] if _bool_var.any() else False
            if res == False:
                self.finalFeature = random.choice(self.featuresNames)
            else:
                temp = copy.deepcopy(self.featuresNames)
                temp.remove(res)
                self.finalFeature = random.choice(temp)

            newComplexity = self.calculateDataComplexity(column, self.finalFeature)
            if newComplexity > self.complexity:
                self.complexity = newComplexity
                self.finalColumn = copy.deepcopy(column)

    # in + out complexity
    def calculateDataComplexity(self, column, thisfeature):

        return DataComplexity.fisher_complexity(column, self.Train_X, self.Train_Y, self.sel_features, thisfeature)

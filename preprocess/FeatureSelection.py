# -*- coding: utf-8 -*-
# @Time  : 2018/4/16 14:00
# @Author: Hanrui Wang
# @Target:

import os.path
import os
import time
import numpy as np

feature_method = ["svm25", "svm50", "svm75", "forest25", "forest50", "forest75", "bsswss25", "bsswss50", "bsswss75"]


def select_features(filename, Train_X, Train_Y, feature_num):
    fsel = FeatureSel(feature_num)
    cache = filename + "_features_" + ".txt"
    sel_features = fsel.select(Train_X, Train_Y, cache)
    feature_F1 = fsel.process_select_F1(Train_X, Train_Y)
    feature_F2 = fsel.process_select_F2(Train_X, Train_Y)
    feature_F3 = fsel.process_select_F3(Train_X, Train_Y)
    feature_F4 = fsel.process_select_F4(Train_X, Train_Y)
    feature_F5 = fsel.process_select_F5(Train_X, Train_Y)
    feature_F6 = fsel.process_select_F6(Train_X, Train_Y)
    feature_F7 = fsel.process_select_F7(Train_X, Train_Y)
    feature_F8 = fsel.process_select_F8(Train_X, Train_Y)
    feature_F9 = fsel.process_select_F9(Train_X, Train_Y)

    return sel_features, feature_F1, feature_F2, feature_F3, feature_F4, feature_F5, feature_F6, feature_F7, feature_F8, feature_F9


class FeatureSel():
    def __init__(self, f_num):
        self.sel_features = {}
        self.cache_changed = False
        self.f_num = f_num

    def select(self, Train_X, Train_Y, cache_path=None):

        if cache_path and os.access(cache_path, os.F_OK):
            self._load_cache(cache_path)
            # If some new feature selection methods is involved.
            # we have to use them  and save the new selected features.
            # methods with cache loaded will skip the process
            self.process_select(Train_X, Train_Y)
            if self.cache_changed:
                self._save_cache(cache_path)
        else:
            self.process_select(Train_X, Train_Y)
            if cache_path:
                self._save_cache(cache_path)
        return self.fused

    def _save_cache(self, cache_path):
        self.makedir_for_cache(cache_path)
        f = file(cache_path, 'w')
        for (k, v) in self.sel_features.iteritems():
            f.write(str(k))
            f.write("::")
            f.write(str(v.tolist()))
            f.write("\n")

    def _load_cache(self, cache_path):
        f = file(cache_path, 'r')
        for line in f:
            (k, v) = line.split("::")
            self.sel_features[k] = np.asarray(eval(v))

    def makedir_for_cache(self, file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def process_select(self, data, labels):
        self.process_select_svmrfe25(data, labels)
        self.process_select_svmrfe50(data, labels)
        self.process_select_svmrfe75(data, labels)
        self.process_select_forest25(data, labels)
        self.process_select_forest50(data, labels)
        self.process_select_forest75(data, labels)
        self.process_select_bsswss25(data, labels)
        self.process_select_bsswss50(data, labels)
        self.process_select_bsswss75(data, labels)
        return self._fuse_features()

    def _fuse_features(self):

        self.fused = []
        for v in self.sel_features.itervalues():
            self.fused += v.tolist()
        self.fused = np.unique(self.fused)
        return self.fused

    def process_select_svmrfe25(self, data, labels):
        if 'svm25' in self.sel_features:
            return

        self.cache_changed = True

        from sklearn.svm import SVC
        from sklearn.feature_selection import RFE

        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=100, step=2)
        rfe.fit(data, labels)

        #        selected = np.arange(features.shape[1])[rfe.support_]
        selected = np.argsort(rfe.ranking_)
        feature_number = int(self.f_num * 0.25)
        selected = selected[:feature_number]
        self.sel_features['svm25'] = selected
        return selected

    def process_select_svmrfe50(self, data, labels):
        if 'svm50' in self.sel_features:
            return

        self.cache_changed = True

        from sklearn.svm import SVC
        from sklearn.feature_selection import RFE

        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=100, step=2)
        rfe.fit(data, labels)

        #        selected = np.arange(features.shape[1])[rfe.support_]
        selected = np.argsort(rfe.ranking_)
        feature_number = int(self.f_num * 0.5)
        selected = selected[:feature_number]
        self.sel_features['svm50'] = selected
        return selected

    def process_select_svmrfe75(self, data, labels):
        if 'svm75' in self.sel_features:
            return

        self.cache_changed = True

        from sklearn.svm import SVC
        from sklearn.feature_selection import RFE

        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=100, step=2)
        rfe.fit(data, labels)

        # selected = np.arange(features.shape[1])[rfe.support_]
        selected = np.argsort(rfe.ranking_)
        feature_number = int(self.f_num * 0.75)
        selected = selected[:feature_number]
        self.sel_features['svm75'] = selected
        return selected

    def process_select_forest25(self, data, labels):
        if 'forest25' in self.sel_features:
            return

        self.cache_changed = True

        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(random_state=0)
        forest.fit(data, labels)

        selected = np.argsort(forest.feature_importances_)[::-1]
        feature_number = int(self.f_num * 0.25)
        selected = selected[:feature_number]
        self.sel_features["forest25"] = selected
        return selected

    def process_select_forest50(self, data, labels):
        if 'forest50' in self.sel_features:
            return

        self.cache_changed = True

        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(random_state=0)
        forest.fit(data, labels)

        selected = np.argsort(forest.feature_importances_)[::-1]
        feature_number = int(self.f_num * 0.5)
        selected = selected[:feature_number]
        self.sel_features["forest50"] = selected
        return selected

    def process_select_forest75(self, data, labels):
        if 'forest75' in self.sel_features:
            return

        self.cache_changed = True

        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(random_state=0)
        forest.fit(data, labels)

        selected = np.argsort(forest.feature_importances_)[::-1]
        feature_number = int(self.f_num * 0.75)
        selected = selected[:feature_number]
        self.sel_features["forest75"] = selected
        return selected

    def process_select_bsswss25(self, data, labels):
        if 'bsswss25' in self.sel_features:
            return

        self.cache_changed = True

        def bss_wss_value(f, labels):
            names = sorted(set(labels))
            wss, bss = np.array([]), np.array([])
            for name in names:
                f_k = f[labels == name]
                f_m = f_k.mean()
                d_m = (f_m - f.mean()) ** 2
                d_z = (f_k - f_m) ** 2
                bss = np.append(bss, d_m.sum())
                wss = np.append(wss, d_z.sum())
            z, m = bss.sum(), wss.sum()
            bsswss = z / m if m > 0 else 0
            return bsswss

        i = 0
        x, y = [], []
        for f in data.transpose():
            x.append(i)
            y.append(bss_wss_value(f, labels))
        selected = np.argsort(y)[::-1]
        feature_number = int(self.f_num * 0.25)
        selected = selected[:feature_number]

        self.sel_features["bsswss25"] = selected
        return selected

    def process_select_bsswss50(self, data, labels):
        if 'bsswss50' in self.sel_features:
            return

        self.cache_changed = True

        def bss_wss_value(f, labels):
            names = sorted(set(labels))
            wss, bss = np.array([]), np.array([])
            for name in names:
                f_k = f[labels == name]
                f_m = f_k.mean()
                d_m = (f_m - f.mean()) ** 2
                d_z = (f_k - f_m) ** 2
                bss = np.append(bss, d_m.sum())
                wss = np.append(wss, d_z.sum())
            z, m = bss.sum(), wss.sum()
            bsswss = z / m if m > 0 else 0
            return bsswss

        i = 0
        x, y = [], []
        for f in data.transpose():
            x.append(i)
            y.append(bss_wss_value(f, labels))
        selected = np.argsort(y)[::-1]
        feature_number = int(self.f_num * 0.5)
        selected = selected[:feature_number]
        self.sel_features["bsswss50"] = selected
        return selected

    def process_select_bsswss75(self, data, labels):
        if 'bsswss75' in self.sel_features:
            return

        self.cache_changed = True

        def bss_wss_value(f, labels):
            names = sorted(set(labels))
            wss, bss = np.array([]), np.array([])
            for name in names:
                f_k = f[labels == name]
                f_m = f_k.mean()
                d_m = (f_m - f.mean()) ** 2
                d_z = (f_k - f_m) ** 2
                bss = np.append(bss, d_m.sum())
                wss = np.append(wss, d_z.sum())
            z, m = bss.sum(), wss.sum()
            bsswss = z / m if m > 0 else 0
            return bsswss

        i = 0
        x, y = [], []
        for f in data.transpose():
            x.append(i)
            y.append(bss_wss_value(f, labels))
        selected = np.argsort(y)[::-1]
        feature_number = int(self.f_num * 0.75)
        selected = selected[:feature_number]
        self.sel_features["bsswss75"] = selected
        return selected

    def process_select_F1(self, data, labels):
        if 'svm25' in self.sel_features:
            F1 = (self.sel_features['svm25']).tolist()
        return F1

    def process_select_F2(self, data, labels):
        if 'svm50' in self.sel_features:
            F2 = (self.sel_features['svm50']).tolist()
        return F2

    def process_select_F3(self, data, labels):
        if 'svm75' in self.sel_features:
            F3 = (self.sel_features['svm75']).tolist()
        return F3

    def process_select_F4(self, data, labels):
        if 'forest25' in self.sel_features:
            F4 = (self.sel_features['forest25']).tolist()
        return F4

    def process_select_F5(self, data, labels):
        if 'forest50' in self.sel_features:
            F5 = (self.sel_features['forest50']).tolist()
        return F5

    def process_select_F6(self, data, labels):
        if 'forest75' in self.sel_features:
            F6 = (self.sel_features['forest75']).tolist()
        return F6

    def process_select_F7(self, data, labels):
        if 'bsswss25' in self.sel_features:
            F7 = (self.sel_features['bsswss25']).tolist()
        return F7

    def process_select_F8(self, data, labels):
        if 'bsswss50' in self.sel_features:
            F8 = (self.sel_features['bsswss50']).tolist()
        return F8

    def process_select_F9(self, data, labels):
        if 'bsswss75' in self.sel_features:
            F9 = (self.sel_features['bsswss75']).tolist()
        return F9

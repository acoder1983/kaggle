# coding: utf-8
from __future__ import division

import sys
import copy
import time
import unittest as ut
from collections import Counter

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

BASE_CLASSIFY_MODELS = [
    (LogisticRegression(random_state=42),
     {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1', 'l2']}),

    (RandomForestClassifier(random_state=42),
     {'n_estimators': [10, 50, 100, 500], 'max_depth':[None, 3, 5, 8],
      'max_features':[None, 'sqrt']}),

    # (GradientBoostingClassifier(random_state=42),
    #  {'n_estimators': [100, 500], 'max_depth':[3, 5, 8], 'learning_rate':[0.01, 0.1],
    #   'max_features':[None, 'sqrt'],
    #   'subsample':[1.0, 0.8, 0.6]
    #   }),

    # (SVC(probability=True, random_state=42),
    #  {'C': [1., 10., 100], 'kernel':['linear', 'sigmoid', 'rbf', 'poly'], 'gamma':[0.001, 0.01, 0.1, 1.]})

]


def buildBaseModels(models):
    baseModels = []
    for m, p in models:
        keys = sorted(list(p.keys()))
        k_cursors = [[len(p[k]), 0] for k in keys]
        while k_cursors[0][1] < k_cursors[0][0]:
            mo = sklearn.clone(m)
            kv = mo.get_params()
            for i, kc in enumerate(k_cursors):
                kv[keys[i]] = p[keys[i]][kc[1]]
            mo.set_params(**kv)
            baseModels.append(mo)
            for i in range(len(keys) - 1, -1, -1):
                if k_cursors[i][1] < k_cursors[i][0] - 1:
                    k_cursors[i][1] += 1
                    break
                else:
                    if i == 0:
                        k_cursors[i][1] += 1
                    else:
                        k_cursors[i][1] = 0

    return baseModels


class BinaryClassifier(BaseEstimator):

    def __init__(self, base_models, sorted_ensembles=None, best_en_idx=0, logger=None):
        self.base_models = base_models
        self.sorted_ensembles = sorted_ensembles
        self.best_en_idx = best_en_idx
        self.logger = logger

    def get_params(self, deep=True):
        return {
            'base_models': self.base_models,
            'sorted_ensembles': self.sorted_ensembles,
            'best_en_idx': self.best_en_idx
        }

    def auto(self, X, y, cv=3):
        skf = StratifiedKFold(random_state=42, n_splits=cv)
        kfolds = list(skf.split(X, y))
        model_probs = []
        en_scores = []
        model_scores = []
        for i, m in enumerate(self.base_models):
            prob, score = self._calc_model_score(m, X, y, kfolds)
            model_probs.append(prob)
            en_scores.append(score)
            model_scores.append(score)
            self.logger.info(
                'calc score [%d] [%.3f] for %s classes %s' % (i, score, m, m.classes_))

        ensembles = [Counter({i: 1}) for i in range(len(self.base_models))]
        MIN_EPS = 1e-3
        for i, en in enumerate(ensembles):
            while True:
                find_m = False
                for k in range(len(self.base_models)):
                    en_copy = ensembles[i].copy()
                    en_copy.update({k: 1})
                    en_copy_score = self._calc_en_score(
                        en_copy, model_probs, y, kfolds)
                    if en_copy_score - en_scores[i] > MIN_EPS:
                        ensembles[i] = en_copy
                        en_scores[i] = en_copy_score
                        find_m = True
                        break
                if not find_m:
                    break
            # self.logger.info('ensemble [%d] score %.3f' % (i, en_scores[i]))
            sys.stdout.write('.')
        sys.stdout.flush()

        self.logger.info('top ensembles')
        self.sorted_ensembles = sorted(zip(en_scores, ensembles),
                                       key=lambda x: x[0], reverse=True)
        for en in self.sorted_ensembles:
            self.logger.info('%.3f %s' % (en[0], en[1]))
        self.best_en_idx = 0

        self.logger.info('top models')
        self.sorted_base_models = sorted(
            enumerate(model_scores), key=lambda x: x[1], reverse=True)
        for i, s in self.sorted_base_models:
            self.logger.info('%.3f [%d]' % (s, i))

        return self

    def fit(self, X, y):
        self.best_model = [(self.base_models[i].fit(X, y), w)
                           for i, w in self.sorted_ensembles[self.best_en_idx][1].items()]

    def select(self, en_idx):
        self.best_en_idx = en_idx

    def getNthBestBaseModel(self, m_idx):
        return self.base_models[self.sorted_base_models[m_idx][0]]

    def _calc_model_score(self, model, X, y, kfolds):
        probs = []
        score = 0.
        for tr_i, ts_i in kfolds:
            X_train = X[tr_i]
            y_train = y[tr_i]
            X_test = X[ts_i]
            y_test = y[ts_i]
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)
            cls_idx = (prob[:, 0] < prob[:, 1]).astype('int')
            y_pred = np.array([model.classes_[i]
                               for i in cls_idx])
            score += accuracy_score(y_test, y_pred)
            probs.append(prob)
        return probs, score / len(kfolds)

    def _calc_en_score(self, ensemble, model_probs, y, kfolds):
        score = 0.
        for i, (_, ts_i) in enumerate(kfolds):
            probs = 0.
            weights = 0
            for m_i, m_c in ensemble.items():
                probs += model_probs[m_i][i] * m_c
                weights += m_c
            avg_prob = probs / weights
            cls_idx = (avg_prob[:, 0] < avg_prob[:, 1]).astype('int')
            y_pred = np.array([self.base_models[0].classes_[i]
                               for i in cls_idx])
            score += accuracy_score(y[ts_i], y_pred)
        return score / len(kfolds)

    def predict(self, X):
        probs = 0.
        weights = 0
        for m, w in self.best_model:
            probs += m.predict_proba(X) * w
            weights += w
        avg_prob = probs / weights
        cls_idx = (avg_prob[:, 0] < avg_prob[:, 1]).astype('int')
        # y_pred = np.array([self.base_models[0].classes_[i] for i in cls_idx])
        y_pred = cls_idx
        return y_pred

    def fit_score(self):
        return self.best_score


class TestAutoClassifier(ut.TestCase):
    def testBuildBaseModels(self):
        models = buildBaseModels([LogisticRegression()],
                                 [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1', 'l2'], 'class_weight':['balanced', None]}])
        # self.assertEqual(2,len(models))
        # self.assertEqual(10, models[1].C)
        self.assertEqual(28, len(models))

    def testFit(self):
        pass


if __name__ == '__main__':
    ut.main()

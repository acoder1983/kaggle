# coding: utf-8
from __future__ import division

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
    (LogisticRegression(random_state=42, n_jobs=1),
     {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1', 'l2'], 'class_weight':['balanced', None]}),

    (RandomForestClassifier(random_state=42),
     {'n_estimators': [10, 50, 100, 500], 'max_depth':[5, 8], 'criterion':['gini', 'entropy']}),

    (GradientBoostingClassifier(random_state=42),
     {'n_estimators': [100, 300, 500, 1000], 'max_depth':[3, 5], 'learning_rate':[0.01, 0.1, 1.0]}),

    (SVC(probability=True),
     {'C': [1., 10., ], 'kernel':['rbf', 'poly'], 'gamma':[0.01, 0.1, 1.], 'coef0':[1., 10., ]})

]


def buildBaseModels(models):
    baseModels = []
    for m, p in models:
        keys = list(p.keys())
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

    def __init__(self):
        self.base_models = buildBaseModels(BASE_CLASSIFY_MODELS)

    def fit(self, X, y):
        skf = StratifiedKFold(random_state=42,n_splits=5)
        kfolds = list(skf.split(X, y))
        model_probs = [self._calc_prob(m, X, y, kfolds)
                       for m in self.base_models]
        ensembles = [Counter({i: 1}) for i in range(len(self.base_models))]
        en_scores = [self._calc_en_score(
            en, model_probs, y, kfolds) for en in ensembles]
        MIN_EPS = 1e-3
        for i, en in enumerate(ensembles):
            print('build ensemble%d ...' % i)
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

        self.best_ens = sorted(zip(en_scores, ensembles),
                               key=lambda x: x[0], reverse=True)
        self.fit_one(0, X, y)
        self._printEn(self.best_ens[:5])

        return self

    def fit_one(self, en_idx, X, y):
        self.best_model = [(self.base_models[i].fit(X, y), w)
                           for i, w in self.best_ens[en_idx][1].items()]

    def _printEn(self, top_ens):
        for te in top_ens:
            print(te)
        top_models = Counter()
        for sc, en in top_ens:
            top_models += en
        for m in sorted(top_models.keys()):
            print(m, self.base_models[m])

    def _calc_prob(self, model, X, y, kfolds):
        print('calc prob for %s' % model.__class__.__name__)
        probs = []
        for tr_i, ts_i in kfolds:
            X_train = X[tr_i]
            y_train = y[tr_i]
            X_test = X[ts_i]
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)
            probs.append(prob)
        return probs

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
        probs = np.zeros((X.shape[0], 2))
        weights = 0
        for m, w in self.best_model:
            probs += m.predict_proba(X) * w
            weights += w
        avg_prob = probs / weights
        cls_idx = (avg_prob[:, 0] < avg_prob[:, 1]).astype('int')
        y_pred = np.array([self.base_models[0].classes_[i] for i in cls_idx])
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

# coding: utf-8
from __future__ import division

import copy
import sys
import time
import unittest as ut
from collections import Counter

import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE_CLASSIFY_MODELS = {
    'lr': (LogisticRegression(random_state=42),
           {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2']}),

    'rf': (RandomForestClassifier(random_state=42),
           {'max_depth': [None, 3, 5, 8, 15],
            'max_features': [None, 'sqrt'],
            'n_estimators': [10, 50, 100, 500], }),

    'gbt': (GradientBoostingClassifier(random_state=42),
            {'learning_rate': [0.01, 0.1],
             'max_depth': [3, 5, 8],
             'max_features': [None, 'sqrt'],
             'n_estimators': [100, 500, 1000],
             'subsample': [1.0, 0.8, 0.6]
             }),

    'svc': (SVC(probability=True, random_state=42),
            {'C': [1., 10.],
             'gamma': [0.001, 0.01, 0.1, 'auto'],
             'kernel': ['linear', 'rbf', 'poly'],
             'coef0': [0., 1., 10., ],
             }),

    'knn': (KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 8],
             'weights': ['uniform', 'distance']})


}


def _build_base_models(labels, meta_models):
    base_models = []
    for l, meta in meta_models.items():
        if l not in labels:
            continue
        m = meta[0]
        p = meta[1]
        keys = sorted(list(p.keys()))
        k_cursors = [[len(p[k]), 0] for k in keys]
        while k_cursors[0][1] < k_cursors[0][0]:
            mo = sklearn.clone(m)
            kv = mo.get_params()
            for i, kc in enumerate(k_cursors):
                kv[keys[i]] = p[keys[i]][kc[1]]
            mo.set_params(**kv)
            base_models.append(mo)
            for i in range(len(keys) - 1, -1, -1):
                if k_cursors[i][1] < k_cursors[i][0] - 1:
                    k_cursors[i][1] += 1
                    break
                else:
                    if i == 0:
                        k_cursors[i][1] += 1
                    else:
                        k_cursors[i][1] = 0

    return base_models


class BinaryClassifier(BaseEstimator):

    def __init__(self, base_models=None, sorted_base_models=None, ensemble=None):
        self.base_models = base_models
        self.sorted_base_models = sorted_base_models
        self.ensemble = ensemble

    def get_params(self, deep=True):
        return {
            'base_models': self.base_models,
            'ensemble': self.ensemble,
        }

    @staticmethod
    def _get_model_idx(model, last_result):
        if len(last_result) > 0:
            p = model.get_params()
            for i, m in enumerate(last_result['base_models']):
                if p == m.get_params():
                    return i
        return -1

    @staticmethod
    def _model_info(model):
        info = model.__class__.__name__ + ' {'
        for _, meta in BASE_CLASSIFY_MODELS.items():
            if meta[0].__class__ == model.__class__:
                pars = sorted(meta[1].keys())
                for p in pars:
                    info += '%s = %s, ' % (p, model.get_params()[p])
                break
        info = info[:len(info) - 2] + '}'
        return info

    @staticmethod
    def hillclimbing(X, y, model_labels=None, last_result={}, cv=3, logger=None):

        base_models = _build_base_models(model_labels, BASE_CLASSIFY_MODELS)

        skf = StratifiedKFold(random_state=42, n_splits=cv)
        kfolds = list(skf.split(X, y))

        # train base models
        model_probs = []
        model_scores = []
        model_scores_std = []
        for i, m in enumerate(base_models):
            logger.info('begin [%d] for %s' %
                        (i, BinaryClassifier._model_info(m)))
            t = time.time()
            m_i = BinaryClassifier._get_model_idx(m, last_result)
            if m_i == -1:
                prob, score, std = BinaryClassifier._calc_model_score(
                    m, X, y, kfolds)
                if len(last_result) > 0:
                    last_result['base_models'].append(m)
                    last_result['model_probs'].append(prob)
                    last_result['model_scores'].append(score)
                    last_result['model_scores_std'].append(std)
            else:
                prob = last_result['model_probs'][m_i]
                score = last_result['model_scores'][m_i]
                std = last_result['model_scores_std'][m_i]

            model_probs.append(prob)
            model_scores.append(score)
            model_scores_std.append(std)
            logger.info(
                'score %.3f time %d' % (score, int(time.time() - t)))

        if len(last_result) == 0:
            last_result['base_models'] = base_models
            last_result['model_probs'] = model_probs
            last_result['model_scores'] = model_scores
            last_result['model_scores_std'] = model_scores_std

        # keep better models
        sorted_base_models = sorted(
            enumerate(model_scores), key=lambda x: x[1], reverse=True)
        keep_model_size = int(len(model_scores) / 5)
        keep_model_idxs = [i for i, _ in list(sorted_base_models)[
            :keep_model_size]]

        # make ensembles
        en_times = 20
        rs = check_random_state(42)
        bag_frac = 0.25
        best_n = 3
        MIN_EPS = 1e-3
        ensemble = Counter()
        for i in range(en_times):
            idxs = rs.permutation(keep_model_idxs)
            cand_idxs = idxs[:int(len(idxs) * bag_frac)]
            ens = Counter(cand_idxs[:best_n])
            en_score, _ = BinaryClassifier._calc_en_score(
                ens, base_models, model_probs, y, kfolds)
            while True:
                find_m = False
                for k in cand_idxs:
                    en_copy = ens.copy()
                    en_copy.update({k: 1})
                    en_copy_score, _ = BinaryClassifier._calc_en_score(
                        en_copy, base_models, model_probs, y, kfolds)
                    if en_copy_score - en_score > MIN_EPS:
                        ens = en_copy
                        en_score = en_copy_score
                        find_m = True
                        break
                if not find_m:
                    break
            ensemble += ens
            logger.info('ensembling %d %s' % (i, ens))

        logger.info('top models')
        for i, s in sorted_base_models:
            logger.info('[%d] %.3f %.3f %s' %
                        (i, s, model_scores_std[i], BinaryClassifier._model_info(base_models[i])))

        en_score, en_score_std = BinaryClassifier._calc_en_score(
            ensemble, base_models, model_probs, y, kfolds)
        logger.info('ensemble result %.3f, %.3f, %s' %
                    (en_score, en_score_std, ensemble))

        return BinaryClassifier(**{'base_models': base_models,
                                   'sorted_base_models': sorted_base_models,
                                   'ensemble': ensemble,
                                   })

    def fit(self, X, y):
        self.best_model = [(self.base_models[i].fit(X, y), w)
                           for i, w in self.ensemble.items()]

    def select(self, en_idx):
        self.best_en_idx = en_idx

    def get_nth_model(self, m_idx):
        return self.base_models[self.sorted_base_models[m_idx][0]]

    @staticmethod
    def _calc_model_score(model, X, y, kfolds):
        probs = []
        score = 0.
        scores = []
        for tr_i, ts_i in kfolds:
            X_train = X.loc[tr_i]
            y_train = y.loc[tr_i]
            X_test = X.loc[ts_i]
            y_test = y.loc[ts_i]
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)
            y_pred = (prob[:, 0] < prob[:, 1]).astype('int')
            # y_pred = np.array([model.classes_[i]
            #                    for i in cls_idx])
            score += accuracy_score(y_test, y_pred)
            scores.append(score)
            probs.append(prob)
        return probs, score / len(kfolds), np.std(scores)

    @staticmethod
    def _calc_en_score(ensemble, base_models, model_probs, y, kfolds):
        score = 0.
        scores = []
        for i, (_, ts_i) in enumerate(kfolds):
            probs = 0.
            weights = 0
            for m_i, m_c in ensemble.items():
                probs += model_probs[m_i][i] * m_c
                weights += m_c
            avg_prob = probs / weights
            y_pred = (avg_prob[:, 0] < avg_prob[:, 1]).astype('int')
            # y_pred = np.array([base_models[0].classes_[i]
            #                    for i in cls_idx])
            score += accuracy_score(y[ts_i], y_pred)
            scores.append(score)
        return score / len(kfolds), np.std(scores)

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


class TestAutoClassifier(ut.TestCase):
    def testBuildBaseModels(self):
        models = _build_base_models([LogisticRegression()],
                                    [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1', 'l2'], 'class_weight':['balanced', None]}])
        # self.assertEqual(2,len(models))
        # self.assertEqual(10, models[1].C)
        self.assertEqual(28, len(models))

    def testFit(self):
        pass


if __name__ == '__main__':
    ut.main()

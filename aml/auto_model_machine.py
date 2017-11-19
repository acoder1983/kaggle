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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

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
             'weights': ['uniform', 'distance']}),

    'mlp': (MLPClassifier(random_state=42),
            {'hidden_layer_sizes': [(50,), (100,), (200,)],
             'activation': ['relu', 'identity', 'logistic', 'tanh']
             }),

    'gau': (GaussianProcessClassifier(random_state=42),
            {'max_iter_predict': [100, 150, 200],

             }),

    'ada': (AdaBoostClassifier(random_state=42),
            {'learning_rate': [0.01, 0.1, 1.],
             'n_estimators': [50, 100, 300, 500],
             }),

    'ext': (ExtraTreesClassifier(random_state=42),
            {'n_estimators': [10, 50, 100, 500],
             'max_features': [None, 'sqrt', 'log2'],
             'max_depth': [3, 5, 8],
             }),

}


class BinaryClassifier(BaseEstimator):

    def __init__(self, base_models, level1_models, cv):
        self.base_models = base_models
        self.level1_models = level1_models
        self.cv = cv

    def get_params(self, deep=True):
        return {
            'base_models': self.base_models,
            'level1_models': self.level1_models,
            'cv': self.cv,
        }

    @staticmethod
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
    def auto(X, y, model_labels=None, last_result={}, cv=5, best_n=1, logger=None):

        base_models = BinaryClassifier._build_base_models(
            model_labels, BASE_CLASSIFY_MODELS)

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

        # choose top base models to build level 1
        model_groups = {}
        for i, m in enumerate(base_models):
            m_name = m.__class__.__name__
            if m_name not in model_groups:
                model_groups[m_name] = []
            model_groups[m_name].append((i, model_scores[i],model_scores_std[i]))
        level1_models = []
        for k, models in model_groups.items():
            level1_models += [m[0] for m in list(
                sorted(model_groups[k], key=lambda x:x[1]-x[2], reverse=True))[:best_n]]

        # use lr as level 2
        # level2_model=

        logger.info('level1 models')
        for i in level1_models:
            logger.info('[%d] %.3f %.3f %s' %
                        (i, model_scores[i], model_scores_std[i], BinaryClassifier._model_info(base_models[i])))

        return BinaryClassifier(**{'base_models': base_models,
                                   'level1_models': level1_models,
                                   'cv': cv,
                                   })

    def fit(self, X, y):
        skf = StratifiedKFold(random_state=42, n_splits=self.cv)
        X_level2 = np.zeros((len(y), len(self.level1_models)))
        for l, m_i in enumerate(self.level1_models):
            y1 = np.array([])
            for tr_i, ts_i in skf.split(X, y):
                model = self.base_models[m_i]
                model.fit(X[tr_i], y[tr_i])
                prob = model.predict_proba(X[ts_i])
                y_pred = (prob[:, 0] < prob[:, 1]).astype('int')
                y1 = np.concatenate([y1, y_pred])
            X_level2[:, l] = y1

        self.level2_model = LogisticRegression()
        # self.level2_model=RandomForestClassifier()
        # self.level2_model=GradientBoostingClassifier()
        self.level2_model.fit(X_level2, y)


    def predict(self, X):
        X_level2 = np.zeros((X.shape[0], len(self.level1_models)))
        for l, m_i in enumerate(self.level1_models):
            model = self.base_models[m_i]
            prob = model.predict_proba(X)
            y_pred = (prob[:, 0] < prob[:, 1]).astype('int')
            X_level2[:, l] = y_pred

        return self.level2_model.predict(X_level2)
    # def get_nth_best_model(self, m_idx):
    #     return self.base_models[self.sorted_base_models[m_idx][0]]

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
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
            score += acc
            probs.append(prob)
        return probs, score / len(kfolds), np.std(scores)


if __name__ == '__main__':
    ut.main()

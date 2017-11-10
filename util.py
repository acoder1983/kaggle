
# coding: utf-8

# In[37]:

import logging
import logging.handlers
import os
import unittest as ut

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class DataFrameSelecter(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.cols].values


class DataFrameSelecter2(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.cols]


class DataFrameSelecter3(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return list(X[self.cols])

# In[ ]:


class DataFrameDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.cols, axis=1)


# In[ ]:

class NanDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.dropna()


def initLogging(log_name):
    LOG_FILE = 'log/%s.log' % log_name
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    handlers = []
    handler = logging.FileHandler(LOG_FILE)  # 实例化handler
    fmt = '%(asctime)s - %(message)s'

    formatter = logging.Formatter(fmt)   # 实例化formatter
    handler.setFormatter(formatter)      # 为handler添加formatter

    logger = logging.getLogger(log_name)    # 获取名为tst的logger
    logger.addHandler(handler)           # 为logger添加handler
    handlers.append(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handlers.append(handler)

    logger.setLevel(logging.DEBUG)
    return logger, handlers


def resetLogging(logger, handlers):
    for h in handlers:
        h.close()
        logger.removeHandler(h)


class Test(ut.TestCase):
    def testDataFrameSelecter(self):
        df = pd.DataFrame({'a': [1, 2], 'b': ['1', '2']})
        Selecter = DataFrameSelecter('a')
        self.assertTrue(np.array_equal(
            np.array([1, 2]), Selecter.transform(df)))

        pipe = Pipeline([('Selecter', DataFrameSelecter('b'))])
        data = pipe.fit_transform(df)
        self.assertTrue(np.array_equal(np.array(['1', '2']), data))


if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

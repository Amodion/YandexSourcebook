'''

Та же история. На машине все работает, но в контесте есть ошибка, но интерпретировать ее не могу, потому что полный лог ошибки не выводит.

'''

from sklearn.base import RegressorMixin
import pandas as pd
import numpy as np

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        df = pd.concat([X, y], axis=1)
        msk_average = df.groupby(['city']).mean().loc['msk'].average_bill
        spb_average = df.groupby(['city']).mean().loc['spb'].average_bill
        self.param = {'msk': msk_average, 'spb': spb_average}

        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        return X['city'].map(self.param).values
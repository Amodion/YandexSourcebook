from sklearn.base import ClassifierMixin
import numpy as np
import pandas as pd

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self.param = X.groupby(['city', 'modified_rubrics'])['average_bill'].median()
        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        return X.apply(lambda x: self.param[x['city'], x['modified_rubrics']], axis=1)
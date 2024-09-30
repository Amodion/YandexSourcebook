from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from typing import Optional, List
import numpy as np
import pandas as pd

class ExponentialLinearRegression(RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.model = Ridge(*args, **kwargs)
    
    def fit(self, X, Y):
        self.model.fit(X, np.log(Y))
        return self

    def predict(self, X):
        return np.exp(self.model.predict(X))

    def get_params(self, *args, **kwargs):
        return self.model.get_params(*args, **kwargs)

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)
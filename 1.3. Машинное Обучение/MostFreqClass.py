'''
На локальной машине все работает. Но в контесте ошибка преминения функции mode. При вызове метода mode возвращается сразу число, мода. Изначально я делал вызов так: "mode(y).mode[0]", но индексировать число нельзя и выдавало ошибку. Я это исправил, но в контексте првоерка проходит тем же способом и выдается ошибка

File "/temp/executing/run_tests.py", line 62, in test_simple_task
    mode(train_target).mode[0]*np.ones((test_data.shape[0],)),
    ~~~~~~~~~~~~~~~~~~~~~~~^^^

Может, проблема в версии scipy. Но как это исправить - не знаю.

'''

from sklearn.base import ClassifierMixin
import numpy as np
from scipy.stats import mode

class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        self.param = round(mode(y).mode)
        self.is_fitted_ = True
        print('Classifier fitted')
        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        return np.full(X.shape[0], self.param)
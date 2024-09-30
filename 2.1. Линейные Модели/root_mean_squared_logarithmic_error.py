'''

Все проходит отлично

'''

import numpy as np

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    assert all(y_true >= 0), 'Отрицательное значение в target! Перепроверьте данные'

    assert y_true.shape[0] == y_pred.shape[0], 'Количество предсказаний не равно количеству таргетов!'

    N = np.int64(y_true.shape[0])

    y_pred = np.array(list(map(lambda x: max(x, a_min), y_pred)))

    return np.sqrt(np.sum((np.log(y_pred) - np.log(y_true))**2) / N)
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
import pandas as pd
import numpy as np


class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.scaler = StandardScaler()

        self.needed_columns = needed_columns

    def fit(self, data, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        if self.needed_columns:
            self.scaler.fit(data[self.needed_columns])
        else:
            self.scaler.fit(data)

        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        if self.needed_columns:
            return self.scaler.transform(data[self.needed_columns])
        else:
            return self.scaler.transform(data)
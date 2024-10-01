'''

Модифицированный пайплайн. Увеличено количество категориальных признаков. Некоторые неприрывные признаки переведы в категориальные.

'''


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import Ridge
from typing import Optional, List
import pandas as pd
import numpy as np

continuous_columns =['Lot_Frontage',
                     'Lot_Area',
                     'Year_Built',
                     'Year_Remod_Add',
                     'Mas_Vnr_Area',
                     'BsmtFin_SF_2',
                     'Bsmt_Unf_SF',
                     'Total_Bsmt_SF',
                     'First_Flr_SF',
                     'Second_Flr_SF',
                     'Gr_Liv_Area',
                     'Bsmt_Full_Bath',
                     'Bsmt_Half_Bath',
                     'Garage_Area',
                     'Wood_Deck_SF',
                     'Open_Porch_SF',
                     'Enclosed_Porch',
                     'Three_season_porch',
                     'Screen_Porch',
                     'Pool_Area',
                     'Misc_Val',
                     'Longitude',
                     'Latitude']

categorial_columns = ['Overall_Qual',
                      'Garage_Qual',
                      'Sale_Condition',
                      'MS_Zoning',
                      'BsmtFin_SF_1',
                      'Low_Qual_Fin_SF',
                      'Full_Bath',
                      'Half_Bath',
                      'Bedroom_AbvGr',
                      'Kitchen_AbvGr',
                      'TotRms_AbvGrd',
                      'Fireplaces',
                      'Garage_Cars',
                      'Mo_Sold',
                      'Year_Sold',
                      'Lot_Shape',
                      'Lot_Config',
                      'Neighborhood',
                      'House_Style',
                      'Overall_Cond',
                      'Exterior_1st',
                      'Exterior_2nd',
                      'Mas_Vnr_Type',
                      'Exter_Qual',
                      'Exter_Cond',
                      'Foundation',
                      'Bsmt_Qual',
                      'Bsmt_Exposure',
                      'BsmtFin_Type_1',
                      'Heating_QC',
                      'Central_Air',
                      'Kitchen_Qual',
                      'Fireplace_Qu',
                      'Garage_Type',
                      'Garage_Finish',
                      'Paved_Drive',
                      'Fence']
                        

class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.scaler = StandardScaler()

        if needed_columns:
            self.needed_columns = needed_columns
        else:
            self.needed_columns = None

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

class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, **kwargs):
        super(OneHotPreprocessor, self).__init__(**kwargs)
        self.Encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
        self.cat = categorial_columns

    def fit(self, data, *args):
        self.Encoder.fit(data[self.cat])
        super().fit(data)
        return self
        
    def transform(self, data):
        data_scale = super().transform(data)
        data_cat = self.Encoder.transform(data[self.cat]).toarray()
        return np.concatenate((data_scale, data_cat), axis=1)

def make_ultimate_pipeline():
    preprocessor = OneHotPreprocessor(needed_columns=continuous_columns)
    estimator = Ridge(alpha=10.0)
    pipe = Pipeline([('Preprocessor', preprocessor), ('Estimator', estimator)])

    return pipe
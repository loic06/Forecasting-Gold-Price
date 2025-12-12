import os
import glob
import joblib
import pickle
import pandas as pd
from forecasting_gold_price.params import *
from sklearn.base import BaseEstimator, TransformerMixin, clone

class TrendEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    - Derive 'Trend' (categorical: 'Up'/'Down') from 'Up Trend' / 'Down Trend'.
    - Derive 'Trend_value' as row-wise sum of the two (skipna), and drop originals.
    - Derive 'Candle_mean' = high - low, and drop originals
    """
    def __init__(self, up_col='Up Trend', down_col='Down Trend', out_cat_col='Trend',
                 out_num_col='Trend_value', high_col='high', low_col='low',
                 out_candle_col='Candle_mean'):
        self.up_col = up_col
        self.down_col = down_col
        self.out_cat_col = out_cat_col
        self.out_num_col = out_num_col
        self.high_col = high_col
        self.low_col = low_col
        self.out_candle_col = out_candle_col

    def fit(self, X, y=None):
        # Nothing to learn
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TrendEngineeringTransformer expects a pandas DataFrame.")

        X = X.copy()
        # 'Trend' = 'Up' if Up Trend not NaN else 'Down'
        X[self.out_cat_col] = np.where(X[self.up_col].notna(), 'Up', 'Down')
        # 'Trend_value' = Up Trend + Down Trend (skipna)
        X[self.out_num_col] = X[[self.up_col, self.down_col]].sum(axis=1, skipna=True)
        # 'Candle_mean' = high - low
        X[self.out_candle_col] = X[self.high_col] - X[self.low_col]

        # Drop originals if requested
        X = X.drop(columns=[self.up_col, self.down_col], errors='ignore')

        return X

    def set_output(self, *, transform=None):
        # No-op to be compatible with pipelines calling set_output
        return self

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drop highly correlated numerical features (> threshold) based on upper triangle of correlation matrix.
    """
    def __init__(self, num_corr_threshold=0.95, method='pearson'):
        self.num_corr_threshold = num_corr_threshold
        self.method = method

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CustomFeatureSelector expects a pandas DataFrame.")
        self.num_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        if len(self.num_cols_) == 0:
            self.num_col_to_drop_ = []
            return self
        corr_num = X[self.num_cols_].corr(method=self.method)
        upper = corr_num.where(np.triu(np.ones(corr_num.shape), k=1).astype(bool)).abs()
        self.num_col_to_drop_ = [c for c in upper.columns if any(upper[c] > self.num_corr_threshold)]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CustomFeatureSelector expects a pandas DataFrame.")
        return X.drop(columns=getattr(self, 'num_col_to_drop_', []), errors='ignore')

    def set_output(self, *, transform=None):
        # No-op to be compatible with pipelines calling set_output
        return self

def load_model() :
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    Return None (but do not Raise) if no model is found

    """
    if MODEL_TARGET == "local":
        #load
        with open('models/champion_model.pkl', 'rb') as f:
            latest_model = pickle.load(f)

        print("✅ Model loaded from local disk")

        return latest_model

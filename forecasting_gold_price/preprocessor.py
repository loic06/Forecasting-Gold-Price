import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class PrePreprocessTransformer(BaseEstimator, TransformerMixin):
    """
    Drop columns mostly fill of 0
    """
    def __init__(self, threshold=0.9, exclude=None):
        self.threshold = threshold
        self.exclude = exclude or []
        self._drop_cols_ = None  # learned set of columns to drop
        self._feature_names_in_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PrePreprocessTransformer expects a pandas DataFrame in fit.")

        self._feature_names_in_ = X.columns.tolist()

        # Work on a copy to avoid mutation
        df = X.copy()

        # Restrict to numeric columns (excluding any explicitly protected ones)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in self.exclude]

        # Compute zero ratio per column using non-NA counts
        zero_ratio = (df[num_cols] == 0).sum() / df[num_cols].count()

        # Columns to drop, learned from training data
        self._drop_cols_ = zero_ratio[zero_ratio >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PrePreprocessTransformer expects a pandas DataFrame in transform.")

        if self._drop_cols_ is None:
            raise RuntimeError("Transformer not fitted. Call fit before transform.")

        # Drop the learned columns (ignore missing ones gracefully)
        return X.drop(columns=[c for c in self._drop_cols_ if c in X.columns], axis=1)

    def get_feature_names_out(self, input_features=None):
        if self._feature_names_in_ is None:
            raise RuntimeError("Transformer not fitted. Call fit before get_feature_names_out.")
        output = [c for c in self._feature_names_in_ if c not in (self._drop_cols_ or [])]
        return np.array(output, dtype=object)

    def set_output(self, *, transform=None):
        # compatibility with sklearn's set_output API
        return self


class CustomPreprocessTransformer(BaseEstimator, TransformerMixin):
    """
     Fill NaN based on a custom strategy:
    - if gap at the begining, fill with the first known value
    - if gap in the middle, linear interpolation
    - if gap at the end, fill with the last known value
    """
    def __init__(self, method='linear', drop_all_nan=False):
        self.method = method
        self.drop_all_nan = drop_all_nan
        self._feature_names_in_ = None
        self._feature_names_out_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreprocessEngineeringTransformer expects a pandas DataFrame in fit.")

        self._feature_names_in_ = X.columns.to_list()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreprocessEngineeringTransformer expects a pandas DataFrame.")

        X = X.copy()

        # Select the columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Fill NaN strategy
        X[numeric_cols] = (
            X[numeric_cols].interpolate(method=self.method, axis=0, limit_direction="both")
        )

        # Drop columns of full NaN
        if self.drop_all_nan:
            all_nan_cols = [c for c in numeric_cols if X[c].isna().all()]
            if all_nan_cols:
                X = X.drop(columns=all_nan_cols)

        # Track output feature names post-transform
        self._feature_names_out_ = X.columns.to_list()

        return X

    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out_ is not None:
            return np.array(self._feature_names_out_, dtype=object)

        if self._feature_names_in_ is not None:
            return np.array(self._feature_names_in_, dtype=object)
        raise RuntimeError("Transformer not fitted. Call fit/transform before get_feature_names_out.")

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
        self.num_cols_ = None
        self.num_col_to_drop_ = None
        self._feature_names_in_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CustomFeatureSelector expects a pandas DataFrame.")
        self._feature_names_in_ = list(X.columns)
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

    def get_feature_names_out(self, input_features=None):
        if self._feature_names_in_ is None:
            raise RuntimeError("Transformer not fitted. Call fit before get_feature_names_out.")
        drop = set(self.num_col_to_drop_ or [])
        output = [c for c in self._feature_names_in_ if c not in drop]
        return np.array(output, dtype=object)

    def set_output(self, *, transform=None):
        # No-op to be compatible with pipelines calling set_output
        return self


def build_preprocessing_pipeline(
    num_corr_threshold=0.95,
    method='linear',
    remove_features=False,
    strategy="custom",
    exclude_for_zero_drop=None
):
    exclude_for_zero_drop = exclude_for_zero_drop or []
    num_selector = make_column_selector(dtype_include=np.number)

    if strategy == "custom":
        pre_steps = Pipeline([
            ("preprocess", PrePreprocessTransformer(exclude=exclude_for_zero_drop)),
            ("imputer", CustomPreprocessTransformer(method=method)),
        ]).set_output(transform="pandas")

        numeric_block = Pipeline([
            ("scaler", RobustScaler())
        ])

        preprocessor = Pipeline([
            ("pre_custom", pre_steps),
            ("ct", ColumnTransformer(
                transformers=[("num", numeric_block, num_selector)],
                remainder="drop"
            ).set_output(transform="pandas"))
        ]).set_output(transform="pandas")

    elif strategy == "mean":
        numeric_block = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_block, num_selector)],
            remainder="drop"
        ).set_output(transform="pandas")

    elif strategy == "median":
        numeric_block = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_block, num_selector)],
            remainder="drop"
        ).set_output(transform="pandas")

    else:
        raise TypeError("Strategy not correct")

    if remove_features:
        pipe_new = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("corr_feature_dropper", CustomFeatureSelector(num_corr_threshold=num_corr_threshold, method='pearson')),
        ]).set_output(transform="pandas")
    else:
        pipe_new = Pipeline(steps=[
            ("preprocessing", preprocessor),
        ]).set_output(transform="pandas")

    print("✅ Preprocessor succesfully")

    return pipe_new

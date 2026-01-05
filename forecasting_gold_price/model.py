import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from forecasting_gold_price.preprocessor import build_preprocessing_pipeline


def initialize_model():
    """
    Initialize the LineareRegression model
    """
    model = LinearRegression()

    print("✅ Model initialized")

    return model


def train_model(
        model,
        X: np.ndarray,
        y: np.ndarray):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    strategy="custom"

    pipe_auto = build_preprocessing_pipeline(remove_features=True, strategy=strategy, exclude_for_zero_drop=['GC_F_Close']) # mean, median

    pipe = Pipeline(steps=[
    ('preprocessing', pipe_auto),
    ('sfm', SelectFromModel(model)),
    ('model', model)])

    model_trained = pipe.fit(X, y)

    print(f"✅ Model trained")

    return model_trained

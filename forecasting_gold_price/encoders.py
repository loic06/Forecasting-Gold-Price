import math
import numpy as np
import pandas as pd
import pygeohash as gh

from taxifare.utils import simple_time_and_memory_tracker

def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    assert isinstance(X, pd.DataFrame)

    #Mettre notre code ici

    return np.stack([hour_sin, hour_cos, dow, month, timedelta], axis=1)

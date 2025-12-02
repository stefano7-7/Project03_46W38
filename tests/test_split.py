"""testing of WindForecast.split() function"""

from project03.wind_forecast import WindForecast
from pathlib import Path
import numpy as np
import pandas as pd


def test_split_basic():
    """
    checking WindForecast.split() for:
    - no NaN 
    - correct dimensions
    """
    # creating testing dataset
    n = 10
    df = pd.DataFrame({
        "Time": pd.date_range("2020-01-01", periods=n, freq="h"),
        "windspeed_100m": np.linspace(5, 15, n),
        "Power": np.linspace(0.1, 0.9, n),
    })

    # applying function to testing dataset
    wf = WindForecast()
    wf.df = df
    wf.split()

    # CHECK 1
    # after shift(-1) and dropna()
    # df2 shall have 1 row less
    total_rows_after_shift = len(wf.y_shifted_train) + len(wf.y_shifted_test)
    assert total_rows_after_shift == len(df) - 1

    # CHECK 2
    # no NaN
    assert not np.isnan(wf.x_train).any()
    assert not np.isnan(wf.x_test).any()
    assert not wf.y_shifted_train.isna().any()
    assert not wf.y_shifted_test.isna().any()
    assert not wf.y_original_test.isna().any()

    # --- x two-dimensional (StandardScaler output) ---
    assert wf.x_train.ndim == 2
    assert wf.x_test.ndim == 2
    assert wf.x_train.shape[1] == 1  # solo windspeed_100m

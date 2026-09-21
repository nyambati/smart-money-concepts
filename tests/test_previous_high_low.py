# The first candle of each period looked up the period *before* the previous
# one, because the period count used a strict "start < candle time" comparison
# and then stepped back two periods.

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc


@pytest.fixture
def three_days_of_hourly_candles():
    """Each day has a flat, distinct high/low so the expected value is obvious.

    day 0 -> high 10, low 0
    day 1 -> high 20, low 10
    day 2 -> high 30, low 20
    """
    index = pd.date_range("2024-01-01", periods=72, freq="1h")
    day = np.arange(72) // 24
    return pd.DataFrame(
        {
            "open": (day + 1) * 10.0 - 5,
            "high": (day + 1) * 10.0,
            "low": day * 10.0,
            "close": (day + 1) * 10.0 - 5,
            "volume": 1.0,
        },
        index=index,
    )


def test_first_candle_of_a_period_sees_the_immediately_previous_period(
    three_days_of_hourly_candles,
):
    result = smc.previous_high_low(three_days_of_hourly_candles, time_frame="1D")
    midnight_of_day_two = pd.Timestamp("2024-01-03 00:00")

    assert result.loc[midnight_of_day_two, "PreviousHigh"] == 20.0
    assert result.loc[midnight_of_day_two, "PreviousLow"] == 10.0


def test_every_candle_in_a_period_sees_the_same_previous_period(
    three_days_of_hourly_candles,
):
    result = smc.previous_high_low(three_days_of_hourly_candles, time_frame="1D")
    day_two = result.loc["2024-01-03"]

    assert day_two["PreviousHigh"].nunique() == 1
    assert day_two["PreviousLow"].nunique() == 1


@pytest.fixture(scope="module")
def eurusd():
    data = pd.read_csv(
        os.path.join(BASE_DIR, "test_data", "EURUSD", "EURUSD_15M.csv")
    ).set_index("Date")
    data.index = pd.to_datetime(data.index)
    data.columns = [c.lower() for c in data.columns]
    return data


@pytest.mark.parametrize("time_frame", ["4h", "1D", "W"])
def test_matches_an_independent_resample_on_real_data(eurusd, time_frame):
    """Cross-check every candle against a plain resample + shift by hand."""
    result = smc.previous_high_low(eurusd, time_frame=time_frame)
    periods = eurusd.resample(time_frame).agg({"high": "max", "low": "min"}).dropna()

    own_period = periods.index.get_indexer(eurusd.index, method="ffill")
    previous = own_period - 1
    expected_high = np.where(previous >= 0, periods["high"].values[previous], np.nan)
    expected_low = np.where(previous >= 0, periods["low"].values[previous], np.nan)

    np.testing.assert_allclose(
        result["PreviousHigh"].values, expected_high, rtol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        result["PreviousLow"].values, expected_low, rtol=1e-6, equal_nan=True
    )

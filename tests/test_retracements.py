# retracements() read direction[i - 1] in the bullish branch but direction[i]
# in the bearish one, so the two disagreed on every bar where a swing flips
# direction. A retracement percentage is sign and scale invariant, so mirroring
# the prices has to mirror the output exactly.

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc

TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data", "EURUSD")


@pytest.fixture(scope="module")
def df():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "EURUSD_15M.csv")).set_index("Date")
    data.index = pd.to_datetime(data.index)
    data.columns = [c.lower() for c in data.columns]
    return data


def mirrored(candles):
    """Flip the chart upside down: every high becomes a low and vice versa."""
    return pd.DataFrame(
        {
            "open": -candles["open"],
            "high": -candles["low"],
            "low": -candles["high"],
            "close": -candles["close"],
            "volume": candles["volume"],
        },
        index=candles.index,
    )


def mirrored_swings(swings):
    """The same swings seen on the flipped chart.

    Mirroring the swing frame rather than re-detecting it keeps this a test of
    retracements alone. swing_highs_lows() has its own small tie-break
    asymmetry - a candle that satisfies both conditions always resolves to a
    swing high - which would otherwise leak in on 8 of the 24424 bars.
    """
    return pd.DataFrame(
        {"HighLow": -swings["HighLow"], "Level": -swings["Level"]},
        index=swings.index,
    )


def test_mirroring_the_chart_flips_the_retracement_direction(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    upright = smc.retracements(df, swings)
    flipped = smc.retracements(mirrored(df), mirrored_swings(swings))

    np.testing.assert_array_equal(
        flipped["Direction"].values, -upright["Direction"].values
    )


def test_retracement_percentages_are_unchanged_by_mirroring(df):
    """The bullish and bearish branches must read the same direction, or a
    direction-flip bar gets a retracement from one branch and not the other."""
    swings = smc.swing_highs_lows(df, swing_length=5)
    upright = smc.retracements(df, swings)
    flipped = smc.retracements(mirrored(df), mirrored_swings(swings))

    np.testing.assert_allclose(
        flipped["CurrentRetracement%"].values,
        upright["CurrentRetracement%"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        flipped["DeepestRetracement%"].values,
        upright["DeepestRetracement%"].values,
        rtol=1e-6,
    )


def test_a_direction_flip_bar_gets_a_retracement_from_either_branch(df):
    """The concrete symptom: bars where the swing flips used to report 0.0 in
    one orientation and a real percentage in the other."""
    swings = smc.swing_highs_lows(df, swing_length=5)
    result = smc.retracements(df, swings)

    direction = result["Direction"].values
    flips = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    settled = flips[(direction[flips] != 0) & (flips > 100)]

    assert len(settled) > 0
    assert np.all(result["CurrentRetracement%"].values[settled] != 0)

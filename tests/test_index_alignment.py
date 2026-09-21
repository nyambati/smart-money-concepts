# Regression tests for issues #67 / #68 / #53:
# every indicator must return a frame aligned to the input dataframe's index,
# so that pd.concat([df, result], axis=1) lines the rows up instead of
# producing a doubled frame full of NaN.

import os
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc

TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data", "EURUSD")


@pytest.fixture(scope="module")
def df():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "EURUSD_15M.csv"))
    data = data.set_index("Date")
    data.index = pd.to_datetime(data.index)
    return data


def test_swing_highs_lows_keeps_the_input_index(df):
    result = smc.swing_highs_lows(df, swing_length=5)
    pd.testing.assert_index_equal(result.index, df.index)


def test_bos_choch_accepts_a_swing_frame_carrying_the_input_index(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    result = smc.bos_choch(df, swings)
    pd.testing.assert_index_equal(result.index, df.index)
    assert result["BOS"].notna().sum() > 0


def test_retracements_accepts_a_swing_frame_carrying_the_input_index(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    result = smc.retracements(df, swings)
    pd.testing.assert_index_equal(result.index, df.index)
    assert (result["Direction"] != 0).sum() > 0


def test_fvg_keeps_the_input_index(df):
    pd.testing.assert_index_equal(smc.fvg(df).index, df.index)


def test_ob_keeps_the_input_index(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    pd.testing.assert_index_equal(smc.ob(df, swings).index, df.index)


def test_liquidity_keeps_the_input_index(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    pd.testing.assert_index_equal(smc.liquidity(df, swings).index, df.index)


def test_previous_high_low_keeps_the_input_index(df):
    pd.testing.assert_index_equal(smc.previous_high_low(df, "1D").index, df.index)


def test_sessions_keeps_the_input_index(df):
    pd.testing.assert_index_equal(smc.sessions(df, "London").index, df.index)


def test_concat_with_the_source_frame_does_not_duplicate_rows(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    joined = pd.concat([df, swings, smc.fvg(df)], axis=1)
    assert len(joined) == len(df)
    assert joined["HighLow"].notna().sum() == swings["HighLow"].notna().sum()

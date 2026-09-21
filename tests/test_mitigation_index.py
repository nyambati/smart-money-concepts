# An unmitigated FVG / order block reported MitigatedIndex == 0, which is
# indistinguishable from "mitigated by the candle at position 0". bos_choch
# already reports NaN for an unbroken level; fvg and ob should match.

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
    return data


def test_unmitigated_fvg_reports_nan_not_zero(df):
    result = smc.fvg(df)
    detected = result["FVG"].notna()

    assert detected.sum() > 0
    assert (result.loc[detected, "MitigatedIndex"] == 0).sum() == 0


def test_mitigated_fvg_still_reports_the_mitigating_position(df):
    result = smc.fvg(df)
    mitigated = result["MitigatedIndex"].notna()

    assert mitigated.sum() > 0
    positions = np.flatnonzero(result["FVG"].notna().values)
    mitigating = result["MitigatedIndex"].values[positions]
    seen = ~np.isnan(mitigating)
    assert np.all(mitigating[seen] > positions[seen])


def test_unmitigated_order_block_reports_nan_not_zero(df):
    swings = smc.swing_highs_lows(df, swing_length=5)
    result = smc.ob(df, swings)
    detected = result["OB"].notna()

    assert detected.sum() > 0
    assert (result.loc[detected, "MitigatedIndex"] == 0).sum() == 0

# ob() reads ohlc["volume"], but the class-wide validator only checks "ohlc",
# so a frame without a volume column raised a bare KeyError from deep inside
# the function instead of the validator's LookupError.

import os
import sys

import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc


@pytest.fixture
def ohlc_without_volume():
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.02, 1.14, 1.24],
        }
    )


def test_ob_reports_a_missing_volume_column(ohlc_without_volume):
    swings = smc.swing_highs_lows(ohlc_without_volume, swing_length=1)
    with pytest.raises(LookupError) as excinfo:
        smc.ob(ohlc_without_volume, swings)
    assert str(excinfo.value) == 'Must have a dataframe column named "volume"'


def test_functions_that_do_not_need_volume_still_work_without_it(ohlc_without_volume):
    swings = smc.swing_highs_lows(ohlc_without_volume, swing_length=1)
    assert len(smc.fvg(ohlc_without_volume)) == len(ohlc_without_volume)
    assert len(smc.bos_choch(ohlc_without_volume, swings)) == len(ohlc_without_volume)

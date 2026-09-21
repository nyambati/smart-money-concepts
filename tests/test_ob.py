# ob() has two branches for picking the order block candle: a scan for the
# extreme between the swing and the break, and a fallback to the candle before
# the break when they are adjacent. The bullish fallback assigned the candle's
# high to Bottom and its low to Top.

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc


def swing_frame(length, highs_lows, levels):
    frame = pd.DataFrame(
        {
            "HighLow": np.full(length, np.nan),
            "Level": np.full(length, np.nan),
        }
    )
    for position, (direction, level) in zip(highs_lows, levels):
        frame.loc[position, "HighLow"] = direction
        frame.loc[position, "Level"] = level
    return frame


@pytest.fixture
def break_on_the_candle_after_the_swing():
    """Swing high of 12 at bar 2, broken by bar 3's close of 12.5.

    The swing and the break are adjacent, so ob() takes the fallback branch.
    """
    return pd.DataFrame(
        {
            "open": [8.0, 9.0, 10.0, 12.0, 12.0, 12.0],
            "high": [10.0, 11.0, 12.0, 13.0, 12.5, 12.5],
            "low": [5.0, 6.0, 7.0, 8.0, 9.0, 9.0],
            "close": [8.0, 9.0, 10.0, 12.5, 12.0, 12.0],
            "volume": [1.0] * 6,
        }
    )


def test_bullish_order_block_top_is_above_its_bottom(break_on_the_candle_after_the_swing):
    swings = swing_frame(6, [2], [(1, 12.0)])
    result = smc.ob(break_on_the_candle_after_the_swing, swings)

    detected = result["OB"].notna()
    assert detected.sum() == 1
    assert (result.loc[detected, "Top"] >= result.loc[detected, "Bottom"]).all()


def test_bullish_fallback_order_block_spans_the_candle_before_the_break(
    break_on_the_candle_after_the_swing,
):
    swings = swing_frame(6, [2], [(1, 12.0)])
    result = smc.ob(break_on_the_candle_after_the_swing, swings)

    assert result["OB"].values[2] == 1
    assert result["Top"].values[2] == 12.0
    assert result["Bottom"].values[2] == 7.0

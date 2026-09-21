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


@pytest.fixture
def bullish_block_then_mitigation():
    """Swing high of 12 at bar 2, broken at bar 3, mitigated by bar 6.

    Bar 6 is the candle whose low trades through the block's bottom.
    """
    return pd.DataFrame(
        {
            "open": [8.0, 9.0, 10.0, 12.0, 11.0, 11.0, 11.0, 11.0],
            "high": [10.0, 11.0, 12.0, 13.0, 11.5, 11.5, 11.5, 11.5],
            "low": [5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 6.5, 6.5],
            "close": [8.0, 9.0, 10.0, 12.5, 11.0, 11.0, 11.0, 11.0],
            "volume": [1.0] * 8,
        }
    )


def test_bullish_mitigation_reports_the_candle_that_broke_the_block(
    bullish_block_then_mitigation,
):
    swings = swing_frame(8, [2], [(1, 12.0)])
    result = smc.ob(bullish_block_then_mitigation, swings)

    assert result["OB"].values[2] == 1
    assert result["Bottom"].values[2] == 7.0
    # bar 6 is the first candle whose low (6.5) trades below the bottom (7.0)
    assert result["MitigatedIndex"].values[2] == 6


@pytest.fixture
def bearish_block_then_mitigation():
    """Mirror of the bullish case: swing low of 5 at bar 2, broken at bar 3."""
    return pd.DataFrame(
        {
            "open": [12.0, 11.0, 10.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            "high": [15.0, 14.0, 13.0, 12.0, 11.0, 11.0, 13.5, 13.5],
            "low": [10.0, 9.0, 5.0, 4.0, 7.5, 7.5, 7.5, 7.5],
            "close": [12.0, 11.0, 10.0, 4.5, 8.0, 8.0, 8.0, 8.0],
            "volume": [1.0] * 8,
        }
    )


def test_bullish_and_bearish_mitigation_use_the_same_convention(
    bullish_block_then_mitigation, bearish_block_then_mitigation
):
    bullish = smc.ob(bullish_block_then_mitigation, swing_frame(8, [2], [(1, 12.0)]))
    bearish = smc.ob(bearish_block_then_mitigation, swing_frame(8, [2], [(-1, 5.0)]))

    assert bullish["OB"].values[2] == 1
    assert bearish["OB"].values[2] == -1
    # both blocks are broken by their bar 6, so both must report 6
    assert bullish["MitigatedIndex"].values[2] == bearish["MitigatedIndex"].values[2]


@pytest.fixture
def bar_claimed_by_both_passes():
    """Bar 3 is an outside bar: the lowest low of bars 2-3 and the highest high
    of bars 1-5, so the bullish and the bearish scan both select it.

    The two passes share the breaker and mitigated_index arrays, keyed by that
    index, so the bullish pass's breaker flag is still set when the bearish
    pass creates its block, which immediately invalidates it.
    """
    return pd.DataFrame(
        {
            "open": [5.0, 5.0, 5.0, 5.0, 6.0, 11.0, 7.0, 7.0],
            "high": [6.0, 10.0, 7.0, 20.0, 12.0, 12.0, 8.0, 8.0],
            "low": [1.0, 4.0, 3.0, 2.0, 5.0, 6.0, 0.4, 0.4],
            "close": [5.0, 5.0, 5.0, 6.0, 11.0, 7.0, 0.5, 0.5],
            "volume": [1.0] * 8,
        }
    )


def test_the_two_passes_do_not_erase_each_others_order_blocks(
    bar_claimed_by_both_passes,
):
    swings = swing_frame(8, [0, 1], [(-1, 1.0), (1, 10.0)])
    result = smc.ob(bar_claimed_by_both_passes, swings)

    # a break above the swing high at bar 4 and below the swing low at bar 6
    # both happened, so the result cannot be empty
    assert result["OB"].notna().sum() > 0

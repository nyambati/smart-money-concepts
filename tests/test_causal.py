# Issues #101 / #34 / #93.
#
# The default swing detection uses a centered window, so the swing marked at
# bar i is only knowable at bar i + swing_length. causal=True must use past
# data only: whatever the indicator says about bar i has to survive unchanged
# when bar i+1 arrives.

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc

TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data", "EURUSD")
SWING_LENGTH = 5


@pytest.fixture(scope="module")
def df():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "EURUSD_15M.csv")).set_index("Date")
    data.index = pd.to_datetime(data.index)
    data.columns = [c.lower() for c in data.columns]
    return data.iloc[:2000]


def marked(frame, column):
    """Positions and values of the non-NaN entries of a column."""
    values = frame[column].values
    positions = np.flatnonzero(~np.isnan(values))
    return positions, values[positions]


def test_causal_swings_never_change_once_reported(df):
    """Replay the data candle by candle: a swing already reported for bar i
    must still be there, with the same direction, after more candles arrive."""
    full = smc.swing_highs_lows(df, swing_length=SWING_LENGTH, causal=True)

    for cut in range(1200, 1260):
        prefix = smc.swing_highs_lows(
            df.iloc[:cut], swing_length=SWING_LENGTH, causal=True
        )
        np.testing.assert_array_equal(
            prefix["HighLow"].values,
            full["HighLow"].values[:cut],
            err_msg=f"history was rewritten when the data ended at bar {cut}",
        )


def test_causal_swing_levels_never_change_once_reported(df):
    full = smc.swing_highs_lows(df, swing_length=SWING_LENGTH, causal=True)

    for cut in range(1200, 1260):
        prefix = smc.swing_highs_lows(
            df.iloc[:cut], swing_length=SWING_LENGTH, causal=True
        )
        np.testing.assert_allclose(
            prefix["Level"].values,
            full["Level"].values[:cut],
            equal_nan=True,
        )


def test_causal_swing_is_reported_on_the_bar_that_confirms_it(df):
    """A swing high is confirmed swing_length bars after the extreme, so it is
    reported there, carrying the extreme's price rather than that bar's."""
    result = smc.swing_highs_lows(df, swing_length=SWING_LENGTH, causal=True)
    positions, directions = marked(result, "HighLow")
    levels = result["Level"].values

    assert len(positions) > 0
    for position, direction in zip(positions, directions):
        extreme = position - SWING_LENGTH
        assert extreme >= 0
        source = "high" if direction == 1 else "low"
        assert levels[position] == df[source].values[extreme]


def test_causal_swings_alternate(df):
    result = smc.swing_highs_lows(df, swing_length=SWING_LENGTH, causal=True)
    _, directions = marked(result, "HighLow")

    assert len(directions) > 2
    assert np.all(directions[:-1] != directions[1:])


def test_default_swing_detection_is_unchanged(df):
    """causal defaults to False, keeping the existing centered-window output."""
    pd.testing.assert_frame_equal(
        smc.swing_highs_lows(df, swing_length=SWING_LENGTH),
        smc.swing_highs_lows(df, swing_length=SWING_LENGTH, causal=False),
    )


def test_default_swing_detection_does_use_future_data(df):
    """Guard on the documented difference: the default marks the swing on the
    extreme itself, which that bar cannot yet know about."""
    result = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
    positions, directions = marked(result, "HighLow")

    interior = [(p, d) for p, d in zip(positions, directions) if 0 < p < len(df) - 1]
    assert len(interior) > 0
    position, direction = interior[0]
    source = "high" if direction == 1 else "low"
    assert result["Level"].values[position] == df[source].values[position]


def unexplained_ob_losses(candles, swing_length, cuts, causal):
    """Order blocks that vanished for a reason other than the breaker reset.

    An order block is legitimately erased when it has already been mitigated
    and price then trades back through it. Anything else is history being
    rewritten, which is what issue #93 reports.
    """
    losses = []
    previous = None
    for cut in cuts:
        swings = smc.swing_highs_lows(
            candles.iloc[:cut], swing_length=swing_length, causal=causal
        )
        frame = smc.ob(candles.iloc[:cut], swings)
        current = {name: frame[name].values for name in ("OB", "Top", "Bottom", "MitigatedIndex")}
        if previous is not None:
            new_bar = cut - 1
            for position in np.flatnonzero(~np.isnan(previous["OB"])):
                if not np.isnan(current["OB"][position]):
                    continue
                mitigated = not np.isnan(previous["MitigatedIndex"][position])
                if previous["OB"][position] == 1:
                    reset = mitigated and candles["high"].values[new_bar] > previous["Top"][position]
                else:
                    reset = mitigated and candles["low"].values[new_bar] < previous["Bottom"][position]
                if not reset:
                    losses.append((cut, position))
        previous = current
    return losses


def test_order_blocks_built_on_causal_swings_are_not_retracted(df):
    """#93: an order block, once printed, must survive the next candle unless
    price invalidates it."""
    losses = unexplained_ob_losses(
        df, SWING_LENGTH, range(900, 1300), causal=True
    )
    assert losses == []


def test_the_centered_swing_frame_is_what_makes_order_blocks_unstable(df):
    """Guard on the diagnosis: the same replay over the default swing frame
    does retract order blocks, because its dedup pass rewrites recent swings."""
    losses = unexplained_ob_losses(
        df, SWING_LENGTH, range(900, 1300), causal=False
    )
    assert len(losses) > 0


@pytest.fixture
def swing_high_confirmed_late():
    """A swing high of 15 at bar 2, confirmed at bar 4 whose own high is 13.

    Bar 5 closes at 14: above the confirming bar's high but below the swing
    high, so it must not create an order block. Bar 6 closes at 16 and must.
    """
    highs = [10.0, 11.0, 15.0, 12.0, 13.0, 14.5, 16.5, 16.0]
    lows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    closes = [9.0, 10.0, 11.0, 11.0, 12.0, 14.0, 16.0, 15.0]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1.0] * len(highs),
        }
    )


def test_causal_swing_is_reported_two_bars_after_the_extreme(swing_high_confirmed_late):
    swings = smc.swing_highs_lows(swing_high_confirmed_late, swing_length=2, causal=True)

    assert swings["HighLow"].values[4] == 1
    assert swings["Level"].values[4] == 15.0


def test_order_block_breaks_against_the_swing_level_not_the_report_bar(
    swing_high_confirmed_late,
):
    swings = smc.swing_highs_lows(swing_high_confirmed_late, swing_length=2, causal=True)
    result = smc.ob(swing_high_confirmed_late, swings)

    # bar 5 closes at 14: over the report bar's high of 13, under the swing high
    assert result["OB"].notna().sum() == 1
    # the break happens on bar 6, so the order block sits on the candle before it
    assert not np.isnan(result["OB"].values[5])
    assert result["OB"].values[5] == 1

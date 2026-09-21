# Issue #46 and friends. Three separate defects:
#   1. "UTC+5" was mapped to the POSIX zone "Etc/GMT+5", which is UTC-5, so the
#      offset was applied in the wrong direction.
#   2. A tz-aware index raised TypeError: Already tz-aware.
#   3. The returned frame carried the UTC-converted index instead of the
#      caller's own index.

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc


def flat_candles(index):
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )


def active_window(result, index):
    on = np.flatnonzero(result["Active"].values)
    return index[on[0]], index[on[-1]]


def test_positive_utc_offset_shifts_the_session_forward():
    """Candles labelled in UTC+5: the 07:00-16:00 UTC London session lands on
    12:00-21:00 in those labels."""
    index = pd.date_range("2024-01-01", periods=96, freq="15min")
    result = smc.sessions(flat_candles(index), "London", time_zone="UTC+5")

    first, last = active_window(result, index)
    assert (first.hour, first.minute) == (12, 0)
    assert (last.hour, last.minute) == (21, 0)


def test_negative_utc_offset_shifts_the_session_backward():
    index = pd.date_range("2024-01-01", periods=96, freq="15min")
    result = smc.sessions(flat_candles(index), "London", time_zone="UTC-3")

    first, last = active_window(result, index)
    assert (first.hour, first.minute) == (4, 0)
    assert (last.hour, last.minute) == (13, 0)


def test_tz_aware_input_uses_its_own_offset():
    """Asia/Kolkata is UTC+5:30, so the 07:00-16:00 UTC London session lands on
    12:30-21:30 local without the caller passing time_zone at all."""
    index = pd.date_range("2024-01-01", periods=96, freq="15min", tz="Asia/Kolkata")
    result = smc.sessions(flat_candles(index), "London")

    first, last = active_window(result, index)
    assert (first.hour, first.minute) == (12, 30)
    assert (last.hour, last.minute) == (21, 30)


def test_result_keeps_the_callers_index_even_when_converting():
    index = pd.date_range("2024-01-01", periods=96, freq="15min")
    result = smc.sessions(flat_candles(index), "London", time_zone="UTC+5")

    pd.testing.assert_index_equal(result.index, index)


def test_inactive_candles_have_no_session_high_or_low():
    index = pd.date_range("2024-01-01", periods=96, freq="15min")
    candles = flat_candles(index)
    candles["high"] = np.arange(96, dtype=float)
    candles["low"] = np.arange(96, dtype=float)
    result = smc.sessions(candles, "London")

    inactive = result["Active"] == 0
    assert inactive.sum() > 0
    assert result.loc[inactive, "High"].isna().all()
    assert result.loc[inactive, "Low"].isna().all()


def utc_hours_active(result, index):
    on = np.flatnonzero(result["Active"].values)
    return index[on[0]], index[on[-1]]


def test_session_times_can_be_given_in_a_named_timezone():
    """09:30-16:00 New York is 14:30-21:00 UTC in January (EST, UTC-5)."""
    index = pd.date_range("2024-01-15", periods=96, freq="15min")
    result = smc.sessions(
        flat_candles(index),
        "Custom",
        start_time="09:30",
        end_time="16:00",
        session_time_zone="America/New_York",
    )

    first, last = utc_hours_active(result, index)
    assert (first.hour, first.minute) == (14, 30)
    assert (last.hour, last.minute) == (21, 0)


def test_named_session_timezone_follows_daylight_saving():
    """The same window is 13:30-20:00 UTC in July (EDT, UTC-4)."""
    index = pd.date_range("2024-07-15", periods=96, freq="15min")
    result = smc.sessions(
        flat_candles(index),
        "Custom",
        start_time="09:30",
        end_time="16:00",
        session_time_zone="America/New_York",
    )

    first, last = utc_hours_active(result, index)
    assert (first.hour, first.minute) == (13, 30)
    assert (last.hour, last.minute) == (20, 0)


def test_session_time_zone_defaults_to_utc():
    index = pd.date_range("2024-07-15", periods=96, freq="15min")
    candles = flat_candles(index)

    pd.testing.assert_frame_equal(
        smc.sessions(candles, "London"),
        smc.sessions(candles, "London", session_time_zone="UTC"),
    )

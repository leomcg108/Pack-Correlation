"""Guards on the data pipeline every other feature is built on.

If the CSVs load wrong or the day indices drift, nothing downstream fails
loudly -- it just produces wrong numbers -- so these are the checks worth
having.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from conftest import BARS_PER_DAY, CLOSES, TRADING_DAYS
from findata_extraction import FinDataExtract


def test_pop_data_dict_parses_every_csv(extractor):
    """Bars must land on a sorted timestamp index, with rows intact."""
    assert set(extractor.data) == set(CLOSES)

    frame = extractor.data["ALPHA"]
    assert len(frame) == len(TRADING_DAYS) * BARS_PER_DAY
    assert frame.index[0] == dt.datetime(2022, 3, 21, 9, 30)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert list(frame["Close"][:BARS_PER_DAY]) == CLOSES["ALPHA"]


def test_a_day_slices_out_of_the_index(extractor):
    """Selecting a day by label is what replaced the ticker_dates offsets."""
    frame = extractor.data["ALPHA"]

    assert list(extractor.trading_days()) == [
        dt.datetime(2022, 3, 21),
        dt.datetime(2022, 3, 22),
        dt.datetime(2022, 3, 23),
    ]

    for day in TRADING_DAYS:
        day_slice = frame.loc[str(day)]
        assert len(day_slice) == BARS_PER_DAY
        assert set(day_slice.index.date) == {day}
        assert list(day_slice["Close"]) == CLOSES["ALPHA"]


def test_slice_data_returns_requested_date_range(extractor):
    """A wrong date lookup returns the wrong days rather than raising."""
    sliced = extractor.slice_data("ALPHA", "2022-03-22", "2022-03-23")

    assert len(sliced) == 2 * BARS_PER_DAY
    assert sliced.index[0] == dt.datetime(2022, 3, 22, 9, 30)
    assert sliced.index[-1] == dt.datetime(2022, 3, 23, 9, 35)


def test_download_ticker_data_creates_then_appends(tmp_path):
    """New tickers get a fresh file; known tickers append to the existing one."""
    calls = []

    def fake_downloader(ticker, start, end, interval):
        calls.append(ticker)
        index = pd.date_range(
            dt.datetime.combine(start, dt.time(9, 30)),
            periods=3,
            freq="min",
            name="Datetime",
        )
        return pd.DataFrame(
            {"Open": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0]}, index=index
        )

    fde = FinDataExtract(downloader=fake_downloader)
    fde.set_file_path(str(tmp_path))
    fde.watchlist = ["NEW"]

    fde.download_ticker_data(weeks=1)

    csv_path = tmp_path / "NEW-1m.csv"
    assert csv_path.exists()
    assert calls == ["NEW"] * 4  # an unseen ticker always pulls the full 4 weeks
    new_ticker_rows = len(pd.read_csv(csv_path))
    assert new_ticker_rows == 4 * 2  # 3 bars per week, final bar dropped

    fde.download_ticker_data(weeks=1)

    assert len(calls) == 5  # a known ticker only pulls the weeks asked for
    assert len(pd.read_csv(csv_path)) == new_ticker_rows + 2

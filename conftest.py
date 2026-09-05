"""Shared fixtures: a small synthetic market with known correlations.

Living at the repo root, this file also puts the root on `sys.path` so the
tests can import the modules under test.
"""

from __future__ import annotations

import datetime as dt

import matplotlib
import pytest

matplotlib.use("Agg")  # never open a plot window during a test run

from findata_extraction import FinDataExtract  # noqa: E402

TRADING_DAYS = [dt.date(2022, 3, 21), dt.date(2022, 3, 22), dt.date(2022, 3, 23)]
BARS_PER_DAY = 6

# Close prices, repeated for each trading day. `find_pack_correlation` drops
# each day's final bar, so the correlations below are fixed by the first five
# values of each series and are exact by construction.
CLOSES = {
    "ALPHA": [100, 101, 103, 102, 104, 105],
    "TWIN": [200, 202, 206, 204, 208, 210],  # 2 * ALPHA   -> corr +1.0
    "MIRROR": [150, 149, 147, 148, 146, 145],  # 250 - ALPHA -> corr -1.0
    "MID": [50, 51, 52, 53, 54, 55],  # -> corr +0.9
    "WEAK": [61, 58, 60, 62, 59, 60],  # -> corr -0.2
}
EXPECTED_CORR = {"TWIN": 1.0, "MIRROR": -1.0, "MID": 0.9, "WEAK": -0.2}


def _write_ticker_csv(
    directory,
    ticker: str,
    closes: list[int],
    skip_minutes: set[int] = frozenset(),
    days: list[dt.date] | None = None,
) -> None:
    """Write one "TICKER-1m.csv" in the layout yfinance produces.

    `skip_minutes` omits bars from every day, standing in for a ticker with
    gaps in its intraday data; `days` overrides which dates it trades on.
    """
    rows = ["Datetime,Open,High,Low,Close,Adj Close,Volume"]

    for day in days or TRADING_DAYS:
        market_open = dt.datetime.combine(day, dt.time(9, 30))
        for minute, close in enumerate(closes):
            if minute in skip_minutes:
                continue
            stamp = market_open + dt.timedelta(minutes=minute)
            rows.append(
                f"{stamp:%Y-%m-%d %H:%M:%S},{close},{close + 1},{close - 1},"
                f"{close},{close},1000"
            )

    (directory / f"{ticker}-1m.csv").write_text("\n".join(rows) + "\n")


def _extractor_for(directory) -> FinDataExtract:
    """Load a directory of ticker CSVs into `data`."""
    fde = FinDataExtract()
    fde.set_file_path(str(directory))
    fde.pop_data_dict()

    return fde


@pytest.fixture
def market_dir(tmp_path):
    """A directory of per-ticker CSV files, as `download_ticker_data` leaves it."""
    for ticker, closes in CLOSES.items():
        _write_ticker_csv(tmp_path, ticker, closes)

    return tmp_path


@pytest.fixture
def extractor(market_dir):
    """A FinDataExtract with `data` built from `market_dir`."""
    return _extractor_for(market_dir)


@pytest.fixture
def gapped_extractor(tmp_path):
    """A market whose second ticker is 2 * ALPHA but missing a mid-day bar.

    Every bar GAPPED does have is exactly twice the alpha's, so the only
    correct correlation is 1.0 no matter which minute is missing.
    """
    _write_ticker_csv(tmp_path, "ALPHA", CLOSES["ALPHA"])
    _write_ticker_csv(tmp_path, "GAPPED", CLOSES["TWIN"], skip_minutes={2})

    return _extractor_for(tmp_path)


@pytest.fixture
def disjoint_extractor(tmp_path):
    """A market where no ticker shares a trading day with the alpha."""
    _write_ticker_csv(tmp_path, "ALPHA", CLOSES["ALPHA"])
    _write_ticker_csv(
        tmp_path,
        "OTHER",
        CLOSES["TWIN"],
        days=[dt.date(2022, 4, 11), dt.date(2022, 4, 12)],
    )

    return _extractor_for(tmp_path)

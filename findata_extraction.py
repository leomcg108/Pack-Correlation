"""Tools for downloading and preparing intraday ticker data via yfinance."""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import pickle
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# a full session is 09:30 to 16:00 less the closing bar
FULL_TRADING_DAY_BARS = 389

# Yahoo Finance only serves about 30 days of 1-minute bars, after which a
# gap in the record can never be filled from it again
INTRADAY_HISTORY_DAYS = 30

# how many of the worst tickers to name individually in a data-quality report
WORST_OFFENDERS = 5


def configure_logging(level: int = logging.INFO) -> None:
    """Send this module's progress messages to the console.

    Without it only warnings are shown, which is the right default when
    these modules are imported by other code. Call it once for the running
    commentary during downloads and data checks.
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


class FinDataExtract:
    """Download, update, and slice intraday ticker data stored as CSV files.

    Every frame in `data` is indexed by its bar timestamps, so a day is a
    label slice (`frame.loc["2022-03-21"]`) and the days themselves come
    from the index rather than from a separate table of row offsets.

    `downloader` is the callable used to fetch bars; it defaults to
    `yfinance.download` and can be replaced to source data elsewhere or to
    exercise the pipeline without network access.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        downloader: Callable[..., pd.DataFrame] = yf.download,
    ) -> None:
        self.data = data
        self.file_path = os.getcwd()
        self.watchlist: list[str] | None = None
        self.downloader = downloader

    def __repr__(self) -> str:
        return "FinDataExtraction object"

    def set_file_path(self, file_path: str) -> None:
        self.file_path = file_path

    def pop_watchlist(self, watchlist_path: str | None = None) -> list[str]:
        """Populate and return the watchlist from a CSV file.

        If no path is given, a small default watchlist is returned instead
        of reading from disk.
        """
        if watchlist_path is None:
            self.watchlist = ["SPY", "QQQ", "DIA", "UVXY"]
            return self.watchlist

        with open(watchlist_path, newline="", encoding="utf-8") as file:
            self.watchlist = [row[0] for row in csv.reader(file)]

        self.watchlist.sort()

        return self.watchlist

    def update_1m_28day(
        self, ticker: str, weeks: int = 4, new_ticker: bool = False
    ) -> None:
        """Download `weeks` of 1-minute ticker data and write it to CSV."""
        data_path = os.path.join(self.file_path, ticker)
        total_data = pd.DataFrame()

        for week_offset in range(1, weeks + 1):
            start = dt.date.today() - dt.timedelta(weeks=week_offset)
            end = start + dt.timedelta(days=7)
            data = self.downloader(ticker, start, end, interval="1m")
            data = data.drop(data.tail(1).index)
            total_data = pd.concat([data, total_data])

        logger.info("%s data downloaded from Yahoo Finance", ticker)

        csv_path = f"{data_path}-1m.csv"
        if not new_ticker:
            old_data = pd.read_csv(csv_path, index_col="Datetime")
            total_data = pd.concat([old_data, total_data])

        total_data.to_csv(csv_path)
        logger.info("%s data written to csv file", ticker)

    def download_ticker_data(self, weeks: int = 4, verify: bool = True) -> None:
        """Update up to `weeks` of 1-minute data for every watchlist ticker.

        Existing tickers are updated incrementally; new tickers are
        downloaded in full (up to the 4-week limit Yahoo Finance allows for
        1-minute bars). Assumes CSV filenames follow the pattern
        "TICK-1m.csv".

        Unless `verify` is False the freshly written files are then loaded
        and checked, so gaps are reported while they can still be refilled.
        """
        if self.watchlist is None:
            self.watchlist = self.pop_watchlist()

        # obtain ticker and file names to include any existing
        file_list = {
            file.name.split("-")[0]: file.name
            for file in os.scandir(self.file_path)
            if file.is_file()
        }

        for ticker in self.watchlist:
            if ticker in file_list:
                logger.info("Updating %s", ticker)
                data_path = os.path.join(self.file_path, file_list[ticker])
                file_check = pd.read_csv(data_path, nrows=1)

                # yfinance commonly drops the "Datetime" column name;
                # restore it so the file can be reopened by that label
                if "Datetime" not in file_check.columns:
                    file_check = pd.read_csv(data_path)
                    file_check.rename(
                        columns={"Unnamed: 0": "Datetime"}, inplace=True
                    )
                    file_check.to_csv(data_path, index=False)

                self.update_1m_28day(ticker, weeks)
            else:
                # new ticker: obtain the max allowed 4 weeks of 1m bars
                logger.info("New ticker %s", ticker)
                self.update_1m_28day(ticker, 4, True)

        logger.info("All data downloaded")

        if verify:
            self.pop_data_dict()
            self.report_data_quality()

    def pop_data_dict(self) -> None:
        """Load every ticker CSV in `file_path` into `self.data`.

        Each frame is indexed by its bar timestamps, sorted, with repeated
        timestamps dropped. Files are read in full on every call, so the
        CSVs on disk remain the single source of truth.
        """
        if self.data is None:
            self.data = {}

        for file in os.scandir(self.file_path):
            if not file.is_file():
                continue

            ticker = file.name.split("-")[0]
            logger.debug("Loading %s", ticker)

            frame = pd.read_csv(os.path.join(self.file_path, file.name))

            # yfinance commonly drops the "Datetime" column name
            if "Datetime" not in frame.columns:
                frame = frame.rename(columns={"Unnamed: 0": "Datetime"})

            # the leading 16 characters are "YYYY-MM-DD HH:MM"; taking them
            # discards any timezone offset, keeping every bar naive and
            # directly comparable across tickers
            frame["Datetime"] = pd.to_datetime(
                frame["Datetime"].str[:16], format="%Y-%m-%d %H:%M"
            )

            frame = frame.set_index("Datetime").sort_index()
            frame = frame[~frame.index.duplicated(keep="first")]

            if len(frame) < 2:
                continue

            self.data[ticker] = frame

    def trading_days(self, ticker: str | None = None) -> pd.DatetimeIndex:
        """Return the distinct days a ticker has data for."""
        if ticker is None:
            ticker = next(iter(self.data))

        return self.data[ticker].index.normalize().unique()

    def slice_data(
        self,
        ticker: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Return the rows between two dates, both days included.

        Dates are "YYYY-MM-DD" strings; either end may be omitted to run to
        the start or end of the available data.
        """
        if ticker is None:
            ticker = next(iter(self.data))
        if ticker not in self.data:
            logger.warning("%s not found in data", ticker)
            return None

        logger.debug("Data slice for %s", ticker)

        return self.data[ticker].loc[start_date:end_date]

    def plot_data(
        self,
        ticker: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        plot_series: str = "Close",
    ) -> None:
        """Plot data for a given ticker and datetime range."""
        if ticker is None:
            ticker = next(iter(self.data))
        if ticker not in self.data:
            logger.warning("%s not found in data", ticker)
            return

        series = self.slice_data(ticker, start_date, end_date)

        plt.plot(series[plot_series], label=ticker)
        plt.xlabel("Time")
        plt.ylabel("Volume" if plot_series == "Volume" else "Stock Price (USD)")
        plt.legend()
        logger.debug("%s-data plotted for %s", plot_series, ticker)

    def data_by_date(
        self, ticker: str | None = None
    ) -> dict[dt.date, pd.DataFrame]:
        """Return a dictionary of date -> that day's rows for a ticker."""
        if ticker is None:
            ticker = next(iter(self.data))

        frame = self.data[ticker]

        return {day: group for day, group in frame.groupby(frame.index.date)}

    def downcast_data(self, data: dict[str, pd.DataFrame] | None = None) -> None:
        """Reduce memory usage by downcasting price columns to float32."""
        if data is None:
            data = self.data

        cols = ["Open", "High", "Low", "Close", "Adj Close"]
        for ticker_data in data.values():
            for column in cols:
                ticker_data[column] = ticker_data[column].astype("float32")

    def verify_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        minute_check: bool = False,
    ) -> tuple[dict[str, list[dt.date]], dict[str, list[tuple[dt.date, int]]]]:
        """Check downloaded data against expected market days.

        Returns a dictionary of tickers to lists of missing trading days,
        and a dictionary of tickers to lists of (date, missing_minute_count)
        tuples for days with incomplete data (populated only when
        `minute_check` is True).
        """
        first_frame = self.data[next(iter(self.data))]

        if start_date is None:
            start = first_frame.index[0].date()
        else:
            start = dt.date.fromisoformat(start_date)

        if end_date is None:
            end = first_frame.index[-1].date()
        else:
            end = dt.date.fromisoformat(end_date)

        # Not true market days, only a list of weekdays in the date range
        market_days = []
        for offset in range((end - start).days + 1):
            candidate = start + dt.timedelta(days=offset)
            if candidate.isoweekday() < 6:
                market_days.append(candidate)

        missed_days_ticker = {}
        missed_mins_ticker = {}

        for ticker, frame in self.data.items():
            bars_per_day = frame.groupby(frame.index.date).size()

            missed_days = list(set(market_days) ^ set(bars_per_day.index))
            if missed_days:
                missed_days_ticker[ticker] = missed_days

            if minute_check:
                missed_mins_ticker[ticker] = [
                    (day, FULL_TRADING_DAY_BARS - bars)
                    for day, bars in bars_per_day.items()
                    if bars < FULL_TRADING_DAY_BARS
                ]

        return missed_days_ticker, missed_mins_ticker

    def report_data_quality(
        self, minute_check: bool = True
    ) -> tuple[dict[str, list[dt.date]], dict[str, list[tuple[dt.date, int]]]]:
        """Run `verify_data` and log what it found.

        Only the worst few tickers are named individually, so a watchlist of
        several hundred still produces a readable report. Returns whatever
        `verify_data` returned, for callers that want the detail.
        """
        if not self.data:
            logger.warning("No data loaded, so there is nothing to check")
            return {}, {}

        missed_days, missed_minutes = self.verify_data(minute_check=minute_check)

        if missed_days:
            total = sum(len(days) for days in missed_days.values())
            logger.warning(
                "%d of %d tickers are missing trading days (%d in total)",
                len(missed_days),
                len(self.data),
                total,
            )
            worst = sorted(
                missed_days.items(), key=lambda item: len(item[1]), reverse=True
            )
            for ticker, days in worst[:WORST_OFFENDERS]:
                logger.warning(
                    "  %s: %d missing, earliest %s", ticker, len(days), min(days)
                )
        else:
            logger.info("Every ticker covers the full range of trading days")

        incomplete = {t: gaps for t, gaps in missed_minutes.items() if gaps}
        if incomplete:
            total = sum(len(gaps) for gaps in incomplete.values())
            logger.warning(
                "%d tickers have days short of a full session (%d days in total)",
                len(incomplete),
                total,
            )
            worst = sorted(
                incomplete.items(), key=lambda item: len(item[1]), reverse=True
            )
            for ticker, gaps in worst[:WORST_OFFENDERS]:
                day, missing = max(gaps, key=lambda gap: gap[1])
                logger.warning(
                    "  %s: %d short days, worst %s (%d minutes missing)",
                    ticker,
                    len(gaps),
                    day,
                    missing,
                )
        elif minute_check:
            logger.info("Every trading day holds a full session of bars")

        self._warn_if_stale()

        return missed_days, missed_minutes

    def _warn_if_stale(self) -> None:
        """Warn when the record is old enough that gaps become permanent."""
        latest = max(frame.index[-1] for frame in self.data.values())
        age = (dt.datetime.now() - latest).days

        if age >= INTRADAY_HISTORY_DAYS:
            logger.warning(
                "Most recent bar is %d days old. Yahoo Finance serves only "
                "about %d days of 1-minute data, so anything older than that "
                "can no longer be backfilled",
                age,
                INTRADAY_HISTORY_DAYS,
            )
        else:
            logger.info(
                "Most recent bar is %d days old (%d days before it falls "
                "outside the 1-minute history window)",
                age,
                INTRADAY_HISTORY_DAYS - age,
            )

    def load_pickle(self, pickle_path: str, data_name: str) -> None:
        """Load a previously pickled `data` dictionary from disk."""
        with open(os.path.join(pickle_path, data_name), "rb") as file_in:
            self.data = pickle.load(file_in)

    def save_pickle(self, pickle_path: str, data_name: str) -> None:
        """Pickle `data` to disk for future use."""
        with open(os.path.join(pickle_path, data_name), "wb") as file_out:
            pickle.dump(self.data, file_out)

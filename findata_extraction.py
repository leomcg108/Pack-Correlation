"""Tools for downloading and preparing intraday ticker data via yfinance."""

from __future__ import annotations

import csv
import datetime as dt
import os
import pickle
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


class FinDataExtract:
    """Download, update, and slice intraday ticker data stored as CSV files.

    `downloader` is the callable used to fetch bars; it defaults to
    `yfinance.download` and can be replaced to source data elsewhere or to
    exercise the pipeline without network access.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        ticker_dates: dict[str, list[list[int]]] | None = None,
        downloader: Callable[..., pd.DataFrame] = yf.download,
    ) -> None:
        self.data = data
        self.ticker_dates = ticker_dates
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

        print(f"{ticker} data downloaded from Yahoo Finance")

        csv_path = f"{data_path}-1m.csv"
        if not new_ticker:
            old_data = pd.read_csv(csv_path, index_col="Datetime")
            total_data = pd.concat([old_data, total_data])

        total_data.to_csv(csv_path)
        print(f"{ticker} data written to csv file\n")

    def download_ticker_data(self, weeks: int = 4) -> None:
        """Update up to `weeks` of 1-minute data for every watchlist ticker.

        Existing tickers are updated incrementally; new tickers are
        downloaded in full (up to the 4-week limit Yahoo Finance allows for
        1-minute bars). Assumes CSV filenames follow the pattern
        "TICK-1m.csv".
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
                print(ticker)
                data_path = os.path.join(self.file_path, file_list[ticker])
                file_check = pd.read_csv(data_path, nrows=1)

                # yfinance commonly drops the "Datetime" column name;
                # restore it if missing
                if "Datetime" not in file_check.columns:
                    file_check = pd.read_csv(data_path)
                    file_check.rename(
                        columns={"Unnamed: 0": "Datetime"}, inplace=True
                    )
                    file_check.to_csv(data_path, index=False)

                self.update_1m_28day(ticker, weeks)
            else:
                # new ticker: obtain the max allowed 4 weeks of 1m bars
                print(f"New: {ticker}")
                self.update_1m_28day(ticker, 4, True)

        print("All data downloaded")

    def pop_data_dict(self) -> None:
        """Load all data files into `self.data`, keyed by ticker.

        Converts the Datetime column from strings to real Timestamp
        objects. Only rows more recent than the last processed index are
        added.
        """
        if self.data is None:
            self.data = {}
        if self.ticker_dates is None:
            self.ticker_dates = {}

        # obtain new file list to include any new files/tickers
        file_list = [file for file in os.scandir(self.file_path) if file.is_file()]

        for file in file_list:
            ticker = file.name.split("-")[0]
            new_data_path = os.path.join(self.file_path, file.name)
            print(ticker)

            # determine the most recently updated data
            if ticker in self.ticker_dates:
                recent_index = self.ticker_dates[ticker][-1][-1]
            else:
                recent_index = 0

            new_data = pd.read_csv(new_data_path)[recent_index:]

            if len(new_data) < 2:
                continue

            if ticker in self.data:
                if "Datetime" not in self.data[ticker].columns:
                    new_data.rename(
                        columns={"Unnamed: 0": "Datetime"}, inplace=True
                    )
            elif "Datetime" not in new_data.columns:
                new_data.rename(columns={"Unnamed: 0": "Datetime"}, inplace=True)

            # convert datetime strings to proper datetime objects
            new_data["Datetime"] = new_data["Datetime"].apply(
                lambda x: dt.datetime.strptime(x[:16], "%Y-%m-%d %H:%M")
            )
            new_data = new_data.sort_values(by="Datetime", ignore_index=True)
            new_data = new_data.drop_duplicates(subset=["Datetime"], keep="first")
            new_data = new_data.reset_index(drop=True)

            if ticker in self.data:
                self.data[ticker] = pd.concat([self.data[ticker], new_data])
            else:
                self.data[ticker] = new_data

    def pop_ticker_dates(self) -> None:
        """Populate `self.ticker_dates` with each day's row-index bounds.

        For every ticker, builds a list of [month, day, year, open_index,
        close_index] entries marking where each trading day begins and
        ends (market open 9:30am and close 4:00pm) within that ticker's
        dataframe.
        """
        if self.data is None:
            print(
                "\nNo data supplied: please pass a dictionary of dataframes "
                "as an argument or use the pop_data_dict() function"
            )
            return

        if self.ticker_dates is None:
            self.ticker_dates = {}

        # obtain new file list to include any new files/tickers
        file_list = [file for file in os.scandir(self.file_path) if file.is_file()]

        for file in file_list:
            ticker = file.name.split("-")[0]
            print(ticker)

            # determine the most recently updated data
            if ticker in self.ticker_dates:
                recent_index = self.ticker_dates[ticker][-1][-1]
            else:
                recent_index = 0

            new_data = self.data[ticker][recent_index:]

            if len(new_data) < 2:
                continue

            begin_index = new_data.index[0]
            end_index = new_data.index[-1]
            dates = []
            month = new_data["Datetime"][begin_index].month
            day = new_data["Datetime"][begin_index].day
            year = new_data["Datetime"][begin_index].year
            start = begin_index
            num_days = 0
            dates.append([month, day, year, start])

            # iterate through dataframe and separate into different days
            for i in range(begin_index + 1, end_index):
                if (
                    new_data["Datetime"][i].date()
                    != new_data["Datetime"][i - 1].date()
                ):
                    start = i
                    month = new_data["Datetime"][i].month
                    day = new_data["Datetime"][i].day
                    year = new_data["Datetime"][i].year
                    dates.append([month, day, year, start])
                    dates[num_days].append(i)
                    num_days += 1

            # adding final index value for final day
            dates[-1].append(end_index + 1)

            if ticker in self.ticker_dates:
                self.ticker_dates[ticker].extend(dates)
            else:
                self.ticker_dates[ticker] = dates

    def slice_data(
        self,
        ticker: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Return the data slice between two dates (inclusive) for a ticker."""
        if ticker is None:
            ticker = next(iter(self.data))
        if ticker not in self.data:
            print(f"{ticker} not found in data")
            return None

        if start_date is None:
            start_index = self.ticker_dates[ticker][0][3]
        else:
            year, month, day = start_date.split("-")
            start_key = [int(month), int(day), int(year)]
            start_index = next(
                x[3] for x in self.ticker_dates[ticker] if x[:3] == start_key
            )

        if end_date is None:
            end_index = self.ticker_dates[ticker][-1][4]
        else:
            year, month, day = end_date.split("-")
            end_key = [int(month), int(day), int(year)]
            end_index = next(
                x[4] for x in self.ticker_dates[ticker] if x[:3] == end_key
            )

        data_slice = self.data[ticker][start_index:end_index]
        data_slice = data_slice.reset_index(drop=True)
        print(f"Data slice for {ticker}")

        return data_slice

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
            print(f"{ticker} not found in data")
            return

        series = self.slice_data(ticker, start_date, end_date)

        plt.plot(series[plot_series], label=ticker)
        plt.xlabel("Index (min)")
        plt.ylabel("Volume" if plot_series == "Volume" else "Stock Price (USD)")
        plt.legend()
        print(f"{plot_series}-data plotted for {ticker}")

    def data_by_date(
        self, ticker: str | None = None
    ) -> dict[dt.date, pd.DataFrame]:
        """Return a dictionary of date -> single-day dataframe for a ticker."""
        if ticker is None:
            ticker = next(iter(self.data))

        data_by_day = {}

        for month, day, year, start_index, end_index in self.ticker_dates[ticker]:
            day_data = self.data[ticker][start_index:end_index]
            data_by_day[dt.date(year=year, month=month, day=day)] = day_data

        return data_by_day

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
        first_ticker = next(iter(self.data))

        if start_date is None:
            start = self.data[first_ticker]["Datetime"][0].date()
        else:
            start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()

        if end_date is None:
            end = self.data[first_ticker]["Datetime"].iloc[-1].date()
        else:
            end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

        # Not true market days, only a list of weekdays in the date range
        market_days = []
        for offset in range((end - start).days + 1):
            candidate = start + dt.timedelta(days=offset)
            if candidate.isoweekday() < 6:
                market_days.append(candidate)

        missed_days_ticker = {}
        missed_mins_ticker = {}

        for ticker, ticker_data in self.data.items():
            day_open = {
                ticker_data["Datetime"][entry[3]].date(): entry
                for entry in self.ticker_dates[ticker]
            }

            missed_days = list(set(market_days) ^ set(day_open.keys()))
            if missed_days:
                missed_days_ticker[ticker] = missed_days

            if minute_check:
                missing_minutes = []
                for date, day in day_open.items():
                    len_day_slice = len(ticker_data[day[3] : day[4]])
                    if len_day_slice < 389:
                        missing_minutes.append((date, 389 - len_day_slice))
                missed_mins_ticker[ticker] = missing_minutes

        return missed_days_ticker, missed_mins_ticker

    def load_pickles(
        self, pickle_path: str, data_name: str, ticker_dates_name: str
    ) -> None:
        """Load previously pickled `data` and `ticker_dates` from disk."""
        with open(os.path.join(pickle_path, data_name), "rb") as file_in:
            self.data = pickle.load(file_in)

        with open(os.path.join(pickle_path, ticker_dates_name), "rb") as file_in:
            self.ticker_dates = pickle.load(file_in)

    def save_pickles(
        self, pickle_path: str, data_name: str, ticker_dates_name: str
    ) -> None:
        """Pickle `data` and `ticker_dates` to disk for future use."""
        with open(os.path.join(pickle_path, data_name), "wb") as file_out:
            pickle.dump(self.data, file_out)

        with open(
            os.path.join(pickle_path, ticker_dates_name), "wb"
        ) as file_out:
            pickle.dump(self.ticker_dates, file_out)

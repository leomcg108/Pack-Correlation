"""Read access shared by the extraction and correlation classes."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class TickerDataAccess:
    """Slicing and plotting over a mapping of ticker to price data.

    Subclasses supply `data`, a mapping of ticker to a dataframe indexed by
    bar timestamp, and may override `default_ticker` to change which ticker
    these methods reach for when called without one.
    """

    data: dict[str, pd.DataFrame]

    def default_ticker(self) -> str:
        """Return the ticker used when a method is called without one."""
        return next(iter(self.data))

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
            ticker = self.default_ticker()
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
        """Plot one series for a given ticker over a date range."""
        if ticker is None:
            ticker = self.default_ticker()
        if ticker not in self.data:
            logger.warning("%s not found in data", ticker)
            return

        series = self.slice_data(ticker, start_date, end_date)

        plt.plot(series[plot_series], label=ticker)
        plt.xlabel("Time")
        plt.ylabel("Volume" if plot_series == "Volume" else "Stock Price (USD)")
        plt.legend()
        logger.debug("%s-data plotted for %s", plot_series, ticker)

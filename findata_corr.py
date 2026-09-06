"""Calculate and visualize intraday correlation of a basket of tickers.

https://github.com/leomcg108/Pack-Correlation/
"""

from __future__ import annotations

import datetime as dt
import logging
from statistics import mean, median, median_high, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from findata_access import TickerDataAccess

logger = logging.getLogger(__name__)

CORR_COLUMNS = [
    "Av Corr",
    "Dir Corr",
    "Median Corr",
    "Stdev Corr",
    "Alpha Gain",
    "Beta",
    "Beta Corr",
    "Epsilon",
    "Epsilon Corr",
    "Sigma",
    "Sigma Corr",
    "Omega",
    "Omega Corr",
]


class PackCorrelation(TickerDataAccess):
    """Correlate a basket of assets against a target asset (alpha).

    Builds `corr_date`, a dataframe indexed by trading day holding the
    basket's average correlation together with the max (beta), median
    (epsilon), least correlated (sigma), and most anti-correlated (omega)
    assets. These quantities define the "pack".

    Also builds `dist_date`, the full distribution of correlations for each
    day, keyed by date.

    `data` maps a ticker to a dataframe indexed by bar timestamp, as
    produced by `FinDataExtract.pop_data_dict`.
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self.data = data
        self.alpha = next(iter(data))

    def __repr__(self) -> str:
        return (
            f"Contains a dictionary of {len(self.data)} dataframes with "
            f"{len(self.trading_days())} days and alpha as {self.alpha}"
        )

    def define_alpha(self, alpha: str) -> None:
        if alpha in self.data:
            self.alpha = alpha
        else:
            logger.warning("%s not in data dictionary", alpha)

    def default_ticker(self) -> str:
        """Reach for the alpha, rather than whichever ticker comes first."""
        return self.alpha

    def trading_days(self) -> pd.DatetimeIndex:
        """Return the distinct days the alpha has data for."""
        return self.data[self.alpha].index.normalize().unique()

    @staticmethod
    def _correlate_block(block: pd.DataFrame, alpha: str) -> dict[str, float]:
        """Correlate every column of one day's closes against the alpha.

        Each pairing uses only the minutes where both tickers traded, so a
        member missing bars loses those minutes rather than shifting its
        remaining prices against the alpha. Members covering too little of
        the day, or whose correlation is undefined, are left out.
        """
        block = block[block[alpha].notna()]
        alpha_closes = block[alpha]
        members = block.drop(columns=alpha)
        len_day = len(alpha_closes)

        # a member has to cover most of the alpha's day to be comparable
        overlap = members.notna().sum()
        members = members.loc[:, (overlap > len_day - 10) & (overlap > 1)]

        if members.empty:
            return {}

        values = members.to_numpy(dtype=float)
        alpha_values = alpha_closes.to_numpy(dtype=float)[:, None]
        valid = ~np.isnan(values)

        # Pearson correlation per column, over each column's own valid rows
        with np.errstate(invalid="ignore", divide="ignore"):
            counts = valid.sum(axis=0)
            mean_alpha = np.where(valid, alpha_values, 0.0).sum(axis=0) / counts
            mean_member = np.where(valid, values, 0.0).sum(axis=0) / counts

            dev_alpha = np.where(valid, alpha_values - mean_alpha, 0.0)
            dev_member = np.where(valid, values - mean_member, 0.0)

            correlations = (dev_alpha * dev_member).sum(axis=0) / np.sqrt(
                (dev_alpha**2).sum(axis=0) * (dev_member**2).sum(axis=0)
            )

        return {
            ticker: float(corr)
            for ticker, corr in zip(members.columns, correlations)
            if np.isfinite(corr)
        }

    def find_pack_correlation(
        self,
        start_index: int | None = None,
        end_index: int | None = None,
        plot_av: bool = True,
    ) -> None:
        """Calculate pack correlation and per-day correlation distribution.

        For each day in the range given by `start_index` and `end_index`,
        computes the average/median/stdev correlation of every other ticker
        to `self.alpha`, along with the most (beta), median (epsilon),
        least (sigma), and most anti-correlated (omega) tickers. Days on
        which nothing correlated are left out of `corr_date`.
        """
        # every ticker's closes on one shared time grid, so a day's
        # correlations are a single pass over a (minutes x tickers) block
        closes = pd.DataFrame(
            {ticker: frame["Close"] for ticker, frame in self.data.items()}
        ).sort_index()

        alpha_frame = self.data[self.alpha]
        self.dist_date = {}
        rows, index = [], []

        for day in self.trading_days()[start_index:end_index]:
            key = str(day.date())

            # the day's final bar sets the alpha's gain but is not correlated
            ticker_corr = self._correlate_block(closes.loc[key].iloc[:-1], self.alpha)
            corr_list = list(ticker_corr.values())
            self.dist_date[day.date()] = corr_list

            # nothing correlated on this day, so there is no pack to describe
            if not ticker_corr:
                continue

            alpha_day = alpha_frame.loc[key]
            alpha_gain = alpha_day["Close"].iloc[-1] / alpha_day["Open"].iloc[0]
            direction = 1 if alpha_gain > 1 else -1

            day_corr = mean(corr_list)
            median_corr = median_high(corr_list)

            beta = max(ticker_corr, key=ticker_corr.get)
            epsilon = next(k for k, v in ticker_corr.items() if v == median_corr)
            omega = min(ticker_corr, key=ticker_corr.get)

            magnitudes = {t: abs(v) for t, v in ticker_corr.items() if v != 0}
            if not magnitudes:  # every correlation was exactly zero
                magnitudes = dict.fromkeys(ticker_corr, 0.0)
            sigma = min(magnitudes, key=magnitudes.get)

            index.append(day)
            rows.append(
                {
                    "Av Corr": day_corr,
                    "Dir Corr": day_corr * direction if day_corr > 0 else 0,
                    "Median Corr": median_corr,
                    "Stdev Corr": (
                        stdev(corr_list) if len(corr_list) > 1 else float("nan")
                    ),
                    "Alpha Gain": alpha_gain,
                    "Beta": beta,
                    "Beta Corr": ticker_corr[beta],
                    "Epsilon": epsilon,
                    "Epsilon Corr": ticker_corr[epsilon],
                    "Sigma": sigma,
                    "Sigma Corr": ticker_corr[sigma],
                    "Omega": omega,
                    "Omega Corr": ticker_corr[omega],
                }
            )

        self.corr_date = pd.DataFrame(
            rows, index=pd.DatetimeIndex(index, name="Day"), columns=CORR_COLUMNS
        )

        if plot_av:
            self._plot_average()

    def _plot_average(self) -> None:
        """Plot the pack's average correlation under a rolling mean."""
        if len(self.corr_date) <= 20:
            roll = 2
        elif len(self.corr_date) <= 100:
            roll = 3
        elif len(self.corr_date) <= 300:
            roll = 5
        else:
            roll = 10

        plt.plot(
            self.corr_date["Av Corr"],
            color="tab:blue",
            alpha=0.5,
            linewidth=2,
            label="Av Corr",
        )
        plt.plot(
            self.corr_date["Av Corr"].rolling(roll).mean(),
            color="tab:orange",
            linewidth=3,
            label=f"Rolling({roll}) Av Corr",
        )
        plt.xlabel("Days")
        plt.ylabel("Correlation")
        plt.legend()

    def plot_day_corr(
        self,
        date: str | None = None,
        plot_alpha: bool = True,
        plot_beta: bool = False,
        plot_epsilon: bool = False,
        plot_sigma: bool = False,
        plot_omega: bool = False,
    ) -> str | None:
        """Plot normalized pack price action for a given day.

        Uses the `corr_date` dataframe built by `find_pack_correlation`
        (computing it for just this day if not yet available). By default
        only alpha is plotted; set the other flags to add more series.
        """
        days = self.trading_days()
        day = days[-1] if not date else pd.Timestamp(date)

        if not hasattr(self, "corr_date") or day not in self.corr_date.index:
            position = days.get_loc(day)
            self.find_pack_correlation(position, position + 1, plot_av=False)

        if day not in self.corr_date.index:
            return f"{day.date()} has no valid correlation data calculated"

        row = self.corr_date.loc[day]

        print(f"\nSelected day: {day.date()}")
        print(f"Average day correlation: {row['Av Corr']:.2f}\n")

        plot_list = []
        if plot_alpha:
            plot_list.append(self.alpha)
            print(f"Alpha: {self.alpha}")

        for role, wanted in (
            ("Beta", plot_beta),
            ("Epsilon", plot_epsilon),
            ("Sigma", plot_sigma),
            ("Omega", plot_omega),
        ):
            if wanted:
                plot_list.append(row[role])
                print(f"{role}: {row[role]} ({row[f'{role} Corr']:.2f})")

        key = str(day.date())
        for ticker in plot_list:
            close = self.data[ticker].loc[key, "Close"]
            norm_close = (close - close.min()) / (close.max() - close.min())

            plt.plot(norm_close.reset_index(drop=True), label=ticker)
            plt.xlabel("Time after market open (min)")
            plt.ylabel("Normalized Price")
            plt.legend()

        return None

    def plot_hist_corr(
        self, date: str | None = None, bins: int = 100, alpha: float = 1
    ) -> None:
        """Plot a histogram of the correlation distribution for one day."""
        if date is None:
            day = list(self.dist_date)[-1]
        else:
            day = dt.date.fromisoformat(date)

        dists = self.dist_date[day]
        counts, bin_edges, _ = plt.hist(dists, bins=bins, alpha=alpha)
        plt.xlim(left=-1, right=1)
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")

        hist_mode = bin_edges[list(counts).index(max(counts))]

        print(f"\nSelected day: {day}\n")
        print(f"Mean: {round(mean(dists), 2)}")
        print(f"Median: {round(median(dists), 2)}")
        print(f"Mode: {round(hist_mode, 2)}")

    def plot_heatmap(
        self,
        start_index: int | None = None,
        end_index: int | None = None,
        bins: int = 100,
    ) -> None:
        """Plot a heatmap of the correlation distributions across days."""
        range_bins = 2000 // bins
        binning = [x / 1000 for x in range(-1000, 1000, range_bins)]
        heatmap_data = pd.DataFrame(index=binning)

        for day in list(self.dist_date)[start_index:end_index]:
            counts = np.histogram(self.dist_date[day], binning)[0]
            heatmap_data[day] = np.append(counts, 0).tolist()

        heatmap = sns.heatmap(
            heatmap_data,
            xticklabels=False,
            yticklabels=False,
            robust=True,
            cbar_kws={"label": "Frequency"},
        )
        heatmap.set_xlabel("Days")
        heatmap.set_ylabel("Correlation")


"""Calculate and visualize intraday correlation of a basket of tickers.

https://github.com/leomcg108/Pack-Correlation/
"""

from __future__ import annotations

from math import isnan
from statistics import mean, median, median_high, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PackCorrelation:
    """Correlate a basket of assets against a target asset (alpha).

    Builds a dataframe of the daily average correlation for the basket,
    the max (beta), median (epsilon), least correlated (sigma), and most
    anti-correlated (omega) assets. These quantities define the "pack".

    Also builds the distribution of correlations for each day as a
    dictionary of lists.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        ticker_dates: dict[str, list[list[int]]],
    ) -> None:
        self.data = data
        self.ticker_dates = ticker_dates
        self.alpha = next(iter(data))

    def __repr__(self) -> str:
        return (
            f"Contains a dictionary of {len(self.data)} dataframes with "
            f"{len(self.ticker_dates[self.alpha])} days and alpha as "
            f"{self.alpha}"
        )

    def define_alpha(self, alpha: str) -> None:
        if alpha in self.data:
            self.alpha = alpha
        else:
            print(f"{alpha} not in data dictionary")

    def find_pack_correlation(
        self,
        start_index: int | None = None,
        end_index: int | None = None,
        plot_av: bool = True,
    ) -> None:
        """Calculate pack correlation and per-day correlation distribution.

        For each day in the range given by `start_index` and `end_index`,
        computes the average/median/stdev correlation of every other
        ticker to `self.alpha`, along with the most (beta), median
        (epsilon), least (sigma), and most anti-correlated (omega)
        tickers. Results are stored in `self.corr_date`; the per-day
        correlation distributions are stored in `self.dist_date`.
        """
        self.dist_date = {}
        alpha_data = self.data[self.alpha]

        self.corr_date = pd.DataFrame(
            columns=[
                "Day", "Av Corr", "Dir Corr", "Median Corr", "Stdev Corr",
                "Alpha Gain", "Beta", "Beta Corr", "Epsilon", "Epsilon Corr",
                "Sigma", "Sigma Corr", "Omega", "Omega Corr",
            ]
        )

        for index_num, date in enumerate(
            self.ticker_dates[self.alpha][start_index:end_index]
        ):
            corr_list = []
            ticker_corr = {}
            day = date[:3]

            alpha_days = [item[:3] for item in self.ticker_dates[self.alpha]]
            alpha_day = alpha_days.index(day)
            alpha_open = self.ticker_dates[self.alpha][alpha_day][3]
            alpha_close = self.ticker_dates[self.alpha][alpha_day][4] - 1
            alpha_slice = alpha_data[alpha_open:alpha_close]
            alpha_gain = (
                alpha_data["Close"][alpha_close] / alpha_data["Open"][alpha_open]
            )

            alpha_slice = alpha_slice["Close"].reset_index(drop=True)
            len_day = len(alpha_slice)

            direction = 1 if alpha_gain > 1 else -1

            for ticker in self.data.keys():
                if ticker == self.alpha:  # avoids self correlation for alpha
                    continue

                all_days = [item[:3] for item in self.ticker_dates[ticker]]
                if day not in all_days:
                    continue

                index_day = all_days.index(day)
                index_open = self.ticker_dates[ticker][index_day][3]
                index_close = self.ticker_dates[ticker][index_day][4] - 1
                day_slice = self.data[ticker][index_open:index_close]
                day_slice = day_slice["Close"].reset_index(drop=True)

                # check that slice is a full day of data
                if len(day_slice) > len_day - 10:
                    corr = alpha_slice.corr(day_slice)
                    corr_list.append(corr)
                    ticker_corr[ticker] = corr

            corr_list = [x for x in corr_list if not isnan(x)]

            day_corr = mean(corr_list)
            median_corr = median_high(corr_list)
            stdev_corr = stdev(corr_list)

            beta = max(ticker_corr, key=ticker_corr.get)
            beta_corr = ticker_corr[beta]

            epsilon = next(k for k, v in ticker_corr.items() if v == median_corr)
            epsilon_corr = ticker_corr[epsilon]

            abs_ticker_corr = {
                key: abs(val) for key, val in ticker_corr.items() if val != 0
            }
            sigma = min(abs_ticker_corr, key=abs_ticker_corr.get)
            sigma_corr = ticker_corr[sigma]

            omega = min(ticker_corr, key=ticker_corr.get)
            omega_corr = ticker_corr[omega]

            self.dist_date[(day[2], day[0], day[1])] = corr_list

            day_corr_dir = day_corr * direction if day_corr > 0 else 0

            if not isnan(day_corr):
                self.corr_date.loc[index_num] = (
                    day, day_corr, day_corr_dir, median_corr, stdev_corr,
                    alpha_gain, beta, beta_corr, epsilon, epsilon_corr,
                    sigma, sigma_corr, omega, omega_corr,
                )

        if plot_av:
            if len(self.corr_date) <= 20:
                roll = 2
            elif len(self.corr_date) <= 100:
                roll = 3
            elif len(self.corr_date) <= 300:
                roll = 5
            else:
                roll = 10

            plt.plot(
                self.corr_date["Av Corr"], color="tab:blue", alpha=0.5,
                linewidth=2, label="Av Corr",
            )
            plt.plot(
                self.corr_date["Av Corr"].rolling(roll).mean(),
                color="tab:orange", linewidth=3,
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
        if not date:
            date = self.ticker_dates[self.alpha][-1][:3]
        else:
            year, month, day = date.split("-")
            date = [int(month), int(day), int(year)]

        if not hasattr(self, "corr_date") or date not in list(
            self.corr_date["Day"]
        ):
            date_index = next(
                i
                for i, x in enumerate(self.ticker_dates[self.alpha])
                if x[:3] == date
            )
            self.find_pack_correlation(
                start_index=date_index, end_index=date_index + 1, plot_av=False
            )

        all_days = list(self.corr_date["Day"])
        date_index = all_days.index(date)

        if self.corr_date["Av Corr"].iloc[date_index] == 0:
            return f"{date} has no valid correlation data calculated"

        alpha = self.alpha
        beta = self.corr_date["Beta"].iloc[date_index]
        epsilon = self.corr_date["Epsilon"].iloc[date_index]
        sigma = self.corr_date["Sigma"].iloc[date_index]
        omega = self.corr_date["Omega"].iloc[date_index]

        av_val = self.corr_date["Av Corr"].iloc[date_index]
        beta_val = self.corr_date["Beta Corr"].iloc[date_index]
        epsilon_val = self.corr_date["Epsilon Corr"].iloc[date_index]
        sigma_val = self.corr_date["Sigma Corr"].iloc[date_index]
        omega_val = self.corr_date["Omega Corr"].iloc[date_index]

        print(f"\nSelected day: {date}")
        print(f"Average day correlation: {av_val:.2f}\n")

        plot_list = []
        if plot_alpha:
            plot_list.append(alpha)
            print(f"Alpha: {alpha}")
        if plot_beta:
            plot_list.append(beta)
            print(f"Beta: {beta} ({beta_val:.2f})")
        if plot_epsilon:
            plot_list.append(epsilon)
            print(f"Epsilon: {epsilon} ({epsilon_val:.2f})")
        if plot_sigma:
            plot_list.append(sigma)
            print(f"Sigma: {sigma} ({sigma_val:.2f})")
        if plot_omega:
            plot_list.append(omega)
            print(f"Omega: {omega} ({omega_val:.2f})")

        for ticker in plot_list:
            date_info = next(x for x in self.ticker_dates[ticker] if x[:3] == date)
            slice_start, slice_end = date_info[3], date_info[4] - 1
            close = self.data[ticker][slice_start:slice_end]["Close"]
            norm_close = (close - close.min()) / (close.max() - close.min())
            norm_close = norm_close.reset_index(drop=True)

            plt.plot(norm_close, label=ticker)
            plt.xlabel("Time after market open (min)")
            plt.ylabel("Normalized Price")
            plt.legend()

        return None

    def plot_hist_corr(
        self, date: str | None = None, bins: int = 100, alpha: float = 1
    ) -> None:
        """Plot a histogram of the correlation distribution for one day."""
        if date is None:
            date = list(self.dist_date.keys())[-1]
        else:
            year, month, day = date.split("-")
            date = (int(year), int(month), int(day))

        dists = self.dist_date[date]
        counts, bin_edges, _ = plt.hist(dists, bins=bins, alpha=alpha)
        plt.xlim(left=-1, right=1)
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")

        hist_mean = mean(dists)
        hist_median = median(dists)
        hist_mode = bin_edges[list(counts).index(max(counts))]

        print(f"\nSelected day: {date}\n")
        print(f"Mean: {round(hist_mean, 2)}")
        print(f"Median: {round(hist_median, 2)}")
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

        for date in list(self.dist_date.keys())[start_index:end_index]:
            counts = np.histogram(self.dist_date[date], binning)[0]
            counts = np.append(counts, 0)
            heatmap_data[date] = counts.tolist()

        heatmap = sns.heatmap(
            heatmap_data,
            xticklabels=False,
            yticklabels=False,
            robust=True,
            cbar_kws={"label": "Frequency"},
        )
        heatmap.set_xlabel("Days")
        heatmap.set_ylabel("Correlation")

    def slice_data(
        self,
        ticker: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Return the data slice between two dates for the given ticker."""
        if ticker is None:
            ticker = self.alpha
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
            end_index = self.ticker_dates[ticker][-1][4] - 1
        else:
            year, month, day = end_date.split("-")
            end_key = [int(month), int(day), int(year)]
            end_index = next(
                x[4] - 1 for x in self.ticker_dates[ticker] if x[:3] == end_key
            )

        data_slice = self.data[ticker][start_index:end_index]
        data_slice = data_slice.reset_index(drop=True)

        print(f"Data slice for {ticker}")

        return data_slice

    def plot_data(
        self,
        ticker: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        plot_series: str = "Close",
    ) -> None:
        """Plot data for a given ticker and datetime range."""
        if ticker is None:
            ticker = self.alpha
        if ticker not in self.data:
            print(f"{ticker} not found in data")
            return

        series = self.slice_data(ticker, start_time, end_time)

        plt.plot(series[plot_series], label=ticker)
        plt.xlabel("Index (min)")
        plt.ylabel("Volume" if plot_series == "Volume" else "Stock Price (USD)")
        plt.legend()

        print(f"{plot_series}-data plotted for {ticker}")

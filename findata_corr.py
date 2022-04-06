# -*- coding: utf-8 -*-
"""

https://github.com/leomcg108/Pack-Correlation/

@author: Leo
"""

import pandas as pd
import matplotlib.pyplot as plt
from statistics import stdev, mean, median, median_high
from math import isnan
import seaborn as sns
import numpy as np
from findata_extraction import FinDataExtract


class PackCorrelation:
    """
    
    Calculates the correlation of assets in a basket to a target asset (alpha) in that
    basket and creates a dataframe of the daily average correlation for the 
    basket, the max (beta), median (epsilon) and least correlated assets (sigma) 
    plus the most anticorrlated asset (omega). These quantities define the "pack".
    
    In addition also creates the distribution of correlations for each day 
    as a dictionary of lists
    """
      
    def __init__(self, data, ticker_dates):
        self.data = data
        self.ticker_dates = ticker_dates
        self.alpha = list(data.keys())[0]

    def __repr__(self):
        
        return f"Contains a dictionary of {len(self.data)} dataframes with {len(self.ticker_dates[self.alpha])} days and alpha as {self.alpha}"
    
    def define_alpha(self, alpha):
        if alpha in self.data:
            self.alpha = alpha
        else:
            print(f"{alpha} not in data dictionary")

    def find_pack_correlation(self, start_index=None, end_index=None, plot_av=True):
        """
        
        Calculates pack correlation and the correlation distribution for each 
        day in the timeframe specified by start_index and end_index
        """
        
        index_num = 0
        self.dist_date = {}
        alpha_data = self.data[self.alpha]

        self.corr_date = pd.DataFrame(columns=["Day", "Av Corr", "Dir Corr",
                                          "Median Corr", "Stdev Corr",
                                          "Alpha Gain", "Beta", "Beta Corr", 
                                          "Epsilon", "Epsilon Corr", "Sigma", 
                                          "Sigma Corr", "Omega", "Omega Corr"])

        for date in self.ticker_dates[self.alpha][start_index:end_index]:
            corr_list = []
            ticker_corr = {}
            day = date[:3]
            # print(day)
            
            alpha_days = [item[:3] for item in self.ticker_dates[self.alpha]]
            alpha_day = alpha_days.index(day)
            alpha_open = self.ticker_dates[self.alpha][alpha_day][3]
            alpha_close = self.ticker_dates[self.alpha][alpha_day][4]-1
            alpha_slice = alpha_data[alpha_open:alpha_close]
            alpha_gain = alpha_data["Close"][alpha_close] \
                          / alpha_data["Open"][alpha_open]
                          
            alpha_slice = alpha_slice["Close"].reset_index(drop=True)
            len_day = len(alpha_slice)
                  
            if alpha_gain > 1:
                direction = 1
            else:
                direction = -1
            
            for ticker in self.data.keys():
                if ticker != self.alpha:  # avoids self correlation for alpha
    
                    all_days = [item[:3] for item in self.ticker_dates[ticker]]
            
                    if day in all_days:
                        index_day = all_days.index(day)
                        index_open = self.ticker_dates[ticker][index_day][3]            
                        index_close = self.ticker_dates[ticker][index_day][4]-1
                        day_slice = self.data[ticker][index_open:index_close]
                        
                        day_slice = day_slice["Close"].reset_index(drop=True)
                        
                        # check that slice is full day of data
                        if len(day_slice) > len_day - 10:
                            corr = alpha_slice.corr(day_slice)                    
                            corr_list.append(corr)
                            ticker_corr[ticker] = corr
                        
            corr_list = [x for x in corr_list if isnan(x)==False]
    
            day_corr = mean(corr_list)
            median_corr = median_high(corr_list)
            stdev_corr = stdev(corr_list)
            
            beta = max(ticker_corr, key=ticker_corr.get)
            beta_corr = ticker_corr[beta]
            
            epsilon = [k for k, v in ticker_corr.items() if v == median_corr][0]
            epsilon_corr = ticker_corr[epsilon]
            
            abs_ticker_corr = {key: abs(val) for key, val in ticker_corr.items() 
                                if val != 0}
            sigma = min(abs_ticker_corr, key=abs_ticker_corr.get)
            sigma_corr = ticker_corr[sigma]
            
            omega = min(ticker_corr, key=ticker_corr.get)
            omega_corr = ticker_corr[omega]

            self.dist_date[(day[2], day[0], day[1])] = corr_list
            
            if day_corr > 0:
                day_corr_dir = day_corr * direction
            else:
                day_corr_dir = 0
            
            if isnan(day_corr) is True:
                pass
            else:
                self.corr_date.loc[index_num] = day, day_corr, day_corr_dir, median_corr, \
                                            stdev_corr, alpha_gain, beta, beta_corr, \
                                            epsilon, epsilon_corr, sigma, sigma_corr, \
                                            omega, omega_corr
            
            index_num += 1
        
        if plot_av == True:   
            if len(self.corr_date) <= 20:
                roll = 2
            elif 20 < len(self.corr_date) <= 100:
                roll = 3
            elif 100 < len(self.corr_date) <= 300:
                roll = 5
            elif len(self.corr_date) > 300:
                roll = 10
    
            plt.plot(self.corr_date["Av Corr"], color="tab:blue", alpha=0.5, 
                     linewidth=2, label="Av Corr")
            plt.plot(self.corr_date["Av Corr"].rolling(roll).mean(), 
                     color="tab:orange", linewidth=3, label=f"Rolling({roll}) Av Corr")
            plt.xlabel("Days")
            plt.ylabel("Correlation")
            plt.legend()
            
        return
            
            
    def plot_day_corr(self, date=None, plot_alpha=True, plot_beta=False,
                  plot_epsilon=False, plot_sigma=False, plot_omega=False):
        """
        
        This function takes the corr_date dataframe calculated with the
        find_pack_correlation method and a date of interest and plots the 
        normalized pack price action for that day. By default only alpha is 
        plotted but argument parameters can be set to True to add plots.
        """

        # use the most recent date as default when no date is specified
        if date is None or not date:
            date = self.ticker_dates[self.alpha][-1][:3]
        else:
            date_list = date.split("-")
            date = [int(date_list[1]), int(date_list[2]), int(date_list[0])]   

        if hasattr(self, "corr_date"):
            all_days = list(self.corr_date["Day"])
            
            if date not in all_days:
                date_index = [self.ticker_dates[self.alpha].index(x) \
                              for x in self.ticker_dates[self.alpha] \
                              if x[:3] == date][0]
                self.find_pack_correlation(start_index=date_index, 
                                       end_index=date_index+1, plot_av=False)

        else:
            date_index = [self.ticker_dates[self.alpha].index(x) \
                          for x in self.ticker_dates[self.alpha] \
                              if x[:3] == date][0]
                
            self.find_pack_correlation(start_index=date_index, 
                                       end_index=date_index+1, plot_av=False)
        
        all_days = list(self.corr_date["Day"])
        
        date_index = all_days.index(date)
        
        if self.corr_date["Av Corr"].iloc[date_index] == 0:
            return f"{date} has no valid correlation data calculated"
        
        a = self.alpha
        b = self.corr_date["Beta"].iloc[date_index]
        e = self.corr_date["Epsilon"].iloc[date_index]
        s = self.corr_date["Sigma"].iloc[date_index]
        o = self.corr_date["Omega"].iloc[date_index]
        
        av_val = self.corr_date["Av Corr"].iloc[date_index]
        b_val = self.corr_date["Beta Corr"].iloc[date_index]
        e_val = self.corr_date["Epsilon Corr"].iloc[date_index]
        s_val = self.corr_date["Sigma Corr"].iloc[date_index]
        o_val = self.corr_date["Omega Corr"].iloc[date_index]
    
        plot_pack = {}  
        plot_list = []
        
        print(f"\nSelected day: {date}")
        print(f"Average day correlation: {av_val:.2f}\n")
        
        if plot_alpha: plot_list.append(a), print(f"Alpha: {a}")
        if plot_beta: plot_list.append(b), print(f"Beta: {b} ({b_val:.2f})")
        if plot_epsilon: plot_list.append(e), print(f"Epsilon: {e} ({e_val:.2f})")
        if plot_sigma: plot_list.append(s), print(f"Sigma: {s} ({s_val:.2f})")
        if plot_omega: plot_list.append(o), print(f"Omega: {o} ({o_val:.2f})")
    
        for t in plot_list:
            date_info = [x for x in self.ticker_dates[t] if x[:3] == date][0]
            slice_start, slice_end = date_info[3], date_info[4]-1
            temp = self.data[t][slice_start:slice_end]
            norm_temp =  (temp["Close"] - temp["Close"].min()) \
                        / (temp["Close"].max() - temp["Close"].min())
            norm_temp.reset_index(drop=True, inplace=True)
            plot_pack[t] = norm_temp
            plt.plot(plot_pack[t], label=t)
            plt.xlabel("Time after market open (min)")
            plt.ylabel("Normalized Price")
            plt.legend()
        
        return


    def plot_hist_corr(self, date=None, bins=100, alpha=1):
        """Plots a histogram of the distribution of correlations for one day"""
        
        if date is None:
            date = list(self.dist_date.keys())[-1]
        else:
            date_list = date.split("-")
            date = (int(date_list[0]), int(date_list[1]), int(date_list[2]))

        dists = self.dist_date[date]
        hist_bins, hist_vals = plt.hist(dists, bins=bins, alpha=alpha)[:-1]
        plt.xlim(left=-1, right=1)
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        plt.legend
        hist_mean = mean(dists)
        hist_median = median(dists)
        hist_mode = hist_vals[list(hist_bins).index(max(hist_bins))]
        print(f"\nSelected day: {date}\n")
        print(f"Mean: {round(hist_mean, 2)}")
        print(f"Median: {round(hist_median, 2)}")
        print(f"Mode: {round(hist_mode, 2)}")
    
        return
    
    
    def plot_heatmap(self, start_index=None, end_index=None, bins=100):
        """Plots a heatmap of the correlation distributions as a function of days"""

        range_bins = 2000 // bins
        binning = [x/1000 for x in range(-1000, 1000, range_bins)]
        heatmap = pd.DataFrame(index=binning)
        
        temp_dates = list(self.dist_date.keys())[start_index:end_index]
        
        for date in temp_dates:
            cut = np.histogram(self.dist_date[date], binning)[0]
            cut = np.append(cut, 0)
            heatmap[date] = cut.tolist()
        
        heatmap = sns.heatmap(heatmap, xticklabels=False, yticklabels=False,
                              robust=True, cbar_kws={"label":"Frequency"})
        heatmap.set_xlabel("Days")
        heatmap.set_ylabel("Correlation")
        
        return


    def slice_data(self, ticker=None, start_date=None, end_date=None):
        """Returns a slice between the given dates for the given ticker"""
        
        if ticker is None:
            ticker = self.alpha
        if ticker not in self.data.keys():
            print(f"{ticker} not found in data")
            
            return None
    
        if start_date is None:
            start_index = self.ticker_dates[ticker][0][3]
        else:
            start = start_date.split("-")
            start_list = [int(start[1]), int(start[2]), int(start[0])]   
            start_index = [x[3] for x in self.ticker_dates[ticker] 
                           if x[:3] == start_list][0]
        
        if end_date is None:
            end_index = self.ticker_dates[ticker][-1][4] - 1
        else:
            end = end_date.split("-")
            end_list = [int(end[1]), int(end[2]), int(end[0])]
            end_index = [x[4]-1 for x in self.ticker_dates[ticker] 
                         if x[:3] == end_list][0]
        
        temp = self.data[ticker][start_index:end_index]
        temp = temp.reset_index(drop=True)
        
        print(f"Data slice for {ticker}")
        
        return temp
    
    
    def plot_data(self, ticker=None, start_time=None, end_time=None,
              plot_series="Close"):
        """Plot data for a given ticker and datetime range"""
    
        if ticker is None:
            ticker = self.alpha
        if ticker not in self.data.keys():
            print(f"{ticker} not found in data")
            return None
        
        plot_data = self.slice_data(ticker, start_time, 
                               end_time)
        
        plt.plot(plot_data[plot_series], label=ticker)
        plt.xlabel("Index (min)")
        if plot_series == "Volume":
            plt.ylabel("Volume")
        else:
            plt.ylabel("Stock Price (USD)")
        plt.legend()
        
        print(f"{plot_series}-data plotted for {ticker}")
        
        return



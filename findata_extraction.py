# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:59:16 2022

@author: Leo
"""

from os import scandir, getcwd, path
import csv
import datetime as dt
import pickle
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class FinDataExtract:
    
    def __init__(self, data=None, ticker_dates=None):
        self.data = data
        self.ticker_dates = ticker_dates
        self.file_path = getcwd()
        self.watchlist = None
        
    def __repr__(self):
        return "FinDataExtraction object"
    
    def set_file_path(self, file_path):
        self.file_path = file_path

    def pop_watchlist(self, watchlist_path=None):
        """populate and return watchlist from csv file at given destination path"""
    
        # default watchlist is returned when no file is specified
        if watchlist_path is None:
            self.watchlist = ["SPY", "QQQ", "DIA", "UVXY"]
            return self.watchlist
        
        with open(watchlist_path) as file:
            self.watchlist = []
            for row in csv.reader(file):   
                self.watchlist.append(row[0])
       
        self.watchlist.sort()
        
        return self.watchlist


    def update_1m_28day(self, ticker, weeks=4, new_ticker=False):
        """Use yfinance to download ticker data and write to csv file"""
        
        data_path = path.join(self.file_path, ticker)
        total_data = pd.DataFrame()
        num_weeks = weeks + 1
        
        for n in range(1, num_weeks):
            start = dt.date.today() - dt.timedelta(weeks=n)
            end = start + dt.timedelta(days=7)
            data = yf.download(ticker, start, end, interval="1m")
            data = data.drop(data.tail(1).index)
            frames = [data, total_data]
            total_data = pd.concat(frames)
            
        print(f"{ticker} data downloaded from Yahoo Finance")
        
        if new_ticker == False:
            old_data = pd.read_csv(f"{data_path}-1m.csv", 
                                   index_col="Datetime")
            update_data = pd.concat([old_data, total_data])
            update_data.to_csv(f"{data_path}-1m.csv")
            print(f"{ticker} data written to csv file\n")
        else:
            total_data.to_csv(f"{data_path}-1m.csv")
            print(f"{ticker} data written to csv file\n")
    
        return


    def download_ticker_data(self, weeks=4):
        """
        
        Download a week's worth of 1m data at a time from Yahoo finance for up 
        to 4 weeks total and update to a csv file. If ticker is new then download
        full 4 weeks and write a new csv file to write_path.
        Assumed filenames of csv files is "TICK-1m.csv""
        """
        
        if self.watchlist is None:
            self.watchlist = self.pop_watchlist()
        
        # obtain ticker and file names to include any existing
        file_list = {file.name.split("-")[0]: file.name for file 
                     in scandir(self.file_path) if file.is_file()}  
        
        for ticker in self.watchlist:
            if ticker in file_list.keys():
                print(ticker)
                fname = file_list[ticker]
                data_path = path.join(self.file_path, fname)
                file_check = pd.read_csv(data_path, nrows=1)
                
                # check for "Datetime" column commonly dropped by yfinance and 
                # replace if unnamed
                if "Datetime" not in file_check.columns:
                    file_check = pd.read_csv(data_path)
                    file_check.rename(columns={"Unnamed: 0":"Datetime"}, 
                                      inplace=True)
                    file_check.to_csv(data_path, index=False)
        
                self.update_1m_28day(ticker, weeks)
               
            else:
                # if new ticker obtain the max allowed 4 weeks for 1m bars
                print(f"New: {ticker}")
                self.update_1m_28day(ticker, 4, True)
               
        print("All data downloaded")
        
        return


    def pop_data_dict(self):
        """
        
        Open all relevant files and populate data dictionary with ticker dataframes
        and change Datetime to real Timestamp object from string.
        """
    
        if self.data is None:
            self.data = {}
        if self.ticker_dates is None:
            self.ticker_dates = {}
            
        # obtain new file list to include any new files/tickers
        file_list = [file for file in scandir(self.file_path) if file.is_file()]
       
        for file in file_list:
            ticker = file.name.split("-")[0]
            new_data_path = path.join(self.file_path, file.name)
            print(ticker)
            
            # determine the most recently updated data
            if ticker in self.ticker_dates.keys():
                recent_index = self.ticker_dates[ticker][-1][-1]
            else:
                recent_index = 0
                
            new_data = pd.read_csv(new_data_path)
            new_data = new_data[recent_index:]
            
            if len(new_data) < 2:
                pass
            else:
                if ticker in self.data.keys():
                    if "Datetime" not in self.data[ticker].columns:
                        new_data.rename(columns={"Unnamed: 0": "Datetime"}, 
                                        inplace=True)
                        
                    # convert datetime strings to proper datetime objects
                    new_data["Datetime"] = new_data["Datetime"].apply(
                                           lambda x: dt.datetime.strptime(
                                                       x[:16], "%Y-%m-%d %H:%M")
                                           )
                    
                    new_data = new_data.sort_values(by="Datetime", ignore_index=True)
                    new_data = new_data.drop_duplicates(subset=["Datetime"], keep="first")
                    new_data = new_data.reset_index(drop=True)
                    self.data[ticker] = pd.concat([self.data[ticker], new_data])
                
                else:
                    if "Datetime" not in new_data.columns:
                        new_data.rename(columns={"Unnamed: 0": "Datetime"}, inplace=True)
                    
                    # convert datetime strings to proper datetime objects
                    new_data["Datetime"] = new_data["Datetime"].apply(
                                           lambda x: dt.datetime.strptime(
                                                       x[:16], "%Y-%m-%d %H:%M")
                                           )
                    
                    new_data = new_data.sort_values(by="Datetime", ignore_index=True)
                    new_data = new_data.drop_duplicates(subset=["Datetime"], keep="first")
                    new_data = new_data.reset_index(drop=True)
                    self.data[ticker] = new_data
            
        return


    def pop_ticker_dates(self):
        """
        
        Populate a dictionary containing a list of lists that holds each 
        month-date-year triplet and corresponding index pairs for market open (9:30am) 
        and close (4:00pm) daily timestamps for the relevant locations in the 
        dataframes.
        """
        
        if self.data is None:
            print("\nNo data supplied: please pass a dictionary of dataframes "
                  + "as an argument or use the pop_data_dict() function")
            return
        
        if self.ticker_dates is None:
            self.ticker_dates = {}
            
        # obtain new file list to include any new files/tickers
        file_list = [file for file in scandir(self.file_path) if file.is_file()]
        
        for file in file_list:
            ticker = file.name.split("-")[0]
            print(ticker)
            
            # determine the most recently updated data
            if ticker in self.ticker_dates.keys():
                recent_index = self.ticker_dates[ticker][-1][-1]
            else:
                recent_index = 0
    
            new_data = self.data[ticker][recent_index:]
            
            if len(new_data) < 2:
                pass
            else:
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
                for i in range(begin_index+1, end_index):
                    if new_data["Datetime"][i].date() != new_data["Datetime"][i-1].date():
                        start = i
                        month = new_data["Datetime"][i].month
                        day = new_data["Datetime"][i].day
                        year = new_data["Datetime"][i].year
                        dates.append([month, day, year, start])
                        dates[num_days].append(i)
                        num_days += 1
            
                # adding final index value for final day
                dates[-1].append(end_index+1)
                
                if ticker in self.ticker_dates.keys():
                    self.ticker_dates[ticker].extend(dates)
                else:
                    self.ticker_dates[ticker] = dates 
        
        return
    
    
    def slice_data(self, ticker=None, start_date=None, end_date=None):
        """Returns a slice between the given dates for the given ticker"""
        
        if ticker is None:
            ticker = list(self.data.keys())[0]
        if ticker not in self.data.keys():
            print(f"{ticker} not found in data")
            return None
    
        if start_date is None:
            start_index = self.ticker_dates[ticker][0][3]
        else:
            start = start_date.split("-")
            start_list = [int(start[1]), int(start[2]), int(start[0])]   
            start_index = [x[3] for x in self.ticker_dates[ticker] if x[:3] == start_list][0]
        
        if end_date is None:
            end_index = self.ticker_dates[ticker][-1][4]
        else:
            end = end_date.split("-")
            end_list = [int(end[1]), int(end[2]), int(end[0])]
            end_index = [x[4] for x in self.ticker_dates[ticker] if x[:3] == end_list][0]
        
        temp = self.data[ticker][start_index:end_index]
        temp = temp.reset_index(drop=True)
        print(f"Data slice for {ticker}")
        
        return temp


    def plot_data(self, ticker=None, start_date=None, end_date=None, plot_series="Close"):
        """Plot data for a given ticker and datetime range"""
        
        if ticker is None:
            ticker = list(self.data.keys())[0]
        if ticker not in self.data.keys():
            print(f"{ticker} not found in data")
            return None
        
        plot_data = self.slice_data(ticker, start_date, end_date)
        
        plt.plot(plot_data[plot_series], label=ticker)
        plt.xlabel("Index (min)")
        if plot_series == "Volume":
            plt.ylabel("Volume")
        else:
            plt.ylabel("Stock Price (USD)")
        plt.legend()
        print(f"{plot_series}-data plotted for {ticker}")
        
        return
    
    
    def data_by_date(self, ticker=None):
        """Accepts a ticker and returns a dictionary of dates and day dataframes"""
        
        if ticker is None:
            ticker = list(self.data.keys())[0]
            
        data_by_day = {}
    
        for date in self.ticker_dates[ticker]:
            start_index = date[3]
            end_index = date[4]
            temp_data = self.data[ticker][start_index:end_index]
            dt_date = dt.date(year=date[2], month=date[0], day=date[1])
        
            data_by_day[dt_date] = temp_data
        
        return data_by_day
    

    def downcast_data(self, data=None):
        """Function to lower memory usage by changing from float64 to float 32"""

        if data is None:
            data = self.data
            
        cols = ["Open", "High", "Low", "Close", "Adj Close"]
        for ticker in data.keys():
            for column in cols:
                data[ticker][column] = data[ticker][column].astype("float32")
        
        return


    def verify_data(self, start_date=None, end_date=None, minute_check=False):
        """
        
        Function accepts start and end dates for which to check the downloaded data
        versus the days that the market is open (weekdays) and returns a dictionary 
        of tickers and corresponding lists of missing days of data plus a dictionary
        of tickers and corresponding list of tuples of dates and number of missing
        minutes of data for those dates if minute_check flag is set to True
        """
        
        if start_date is None:
            start = self.data[list(self.data.keys())[0]]["Datetime"][0].date()
        else:
            start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    
        if end_date is None:
            end = self.data[list(self.data.keys())[0]]["Datetime"].iloc[-1].date()
        else:
            end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Not true market days, only a list of weekdays in the date range
        market_days = []
        
        for x in range((end - start).days+1):    
            next_day = start + dt.timedelta(days=x)
            if next_day.isoweekday() < 6:
                market_days.append(next_day)
        
        missed_days_ticker = {}
        missed_mins_ticker = {}
    
        for ticker in self.data.keys():
            missed_days = []
            missing_minutes = []
    
            day_open = {self.data[ticker]["Datetime"][i[3]].date():i for i in self.ticker_dates[ticker]}
            
            missed_days = list(set(market_days) ^ set(day_open.keys()))
            
            if minute_check == True:
                for date, day in day_open.items():
                    len_day_slice = len(self.data[ticker][day[3]:day[4]])
                    if len_day_slice < 389:
                        missing_minutes.append((date, 389-len_day_slice))
                
                missed_mins_ticker[ticker] = missing_minutes
            
            if len(missed_days) > 0:
                missed_days_ticker[ticker] = missed_days
            if len(missing_minutes) > 0:
                missed_mins_ticker[ticker] = missing_minutes

        return missed_days_ticker, missed_mins_ticker


    def load_pickles(self, pickle_path, data_name, ticker_dates_name):
        """load and return pickle files of previously stored data and ticker_dates"""
        
        data_in = path.join(pickle_path, data_name)
        file_in = open(data_in, "rb")
        data = pickle.load(file_in)
        
        dates_in = path.join(pickle_path, ticker_dates_name)
        file_in = open(dates_in, "rb")
        ticker_dates = pickle.load(file_in)
        
        self.data = data
        self.ticker_dates = ticker_dates
        
        return
    
    
    def save_pickles(self, pickle_path, data_name, ticker_dates_name):
        """Save/pickle data and ticker_dates for future use"""
        
        data_out = path.join(pickle_path, data_name)
        filehandler = open(data_out, "wb")
        pickle.dump(self.data, filehandler)
        filehandler.close()
        
        dates_out = path.join(pickle_path, ticker_dates_name)
        filehandler = open(dates_out, "wb")
        pickle.dump(self.ticker_dates, filehandler)
        filehandler.close()
        
        return



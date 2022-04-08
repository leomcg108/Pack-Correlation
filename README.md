# Pack-Correlation
> A unique way to measure stock market correlation

## Table of Contents
* Motivation
* Features
* Prerequisites
* Examples
* Comparison to simple correlations
* Future Work

## Motivation
This project came out of a desire to characterize the US stock market movements and those days when all stocks moved in the same direction, in the same way; all stocks are essentially “in-sync”. Obviously such days will have a large effect on any trading strategy as a stock’s price movement is dictated by the overall market movement; you may be swimming against the tide. The idea of Pack Correlation may be a suitable metric to gauge this market behaviour. 

>PackCorrelation calculates the correlation of assets in a basket to a target asset (alpha) and creates a dataframe of the daily average correlation for the basket, the max (beta), median (epsilon) and least correlated assets (sigma) plus the most anticorrlated asset (omega). These quantities define the "pack".

Mathematically the average pack correlation is:

<img src="https://latex.codecogs.com/svg.image?\overline{\rho&space;}_{x,\alpha&space;}&space;=&space;\frac{1}{N}\sum_{i}^{N}\rho&space;_{x_{i},\alpha&space;}">

where &alpha; is the stock to which others are compared, <img src="https://latex.codecogs.com/svg.image?x_{i}&space;"> is the <img src="https://latex.codecogs.com/svg.image?i^{th}"> ticker in the pack and N is the size of the pack.

So therefore the other members of the pack can be defined as follows:

<img src="https://latex.codecogs.com/svg.image?\beta&space;=&space;max(\rho&space;_{x_i,&space;\alpha})">
<img src="https://latex.codecogs.com/svg.image?\epsilon&space;=&space;median(\rho&space;_{x_i,&space;\alpha})">

<img src="https://latex.codecogs.com/svg.image?\Omega&space;=&space;min(\rho&space;_{x_i,&space;\alpha})">
<img src="https://latex.codecogs.com/svg.image?\sigma&space;=&space;min(\left|&space;\rho&space;_{x_i,&space;\alpha}\right|)">


## Features
* Calculate an average correlation for group of stocks to a predefined lead/alpha stock for each day under consideration
* Helps characterize a stock’s price movement relative to the alpha e.g. highly correlated, uncorrelated or anticorrelated
* Visualize the distribution of correlations of stocks to the alpha for a given day or the whole date range

## Prerequisites
* Python
* Pandas
* Numpy
* Matplotlib
* Seaborn
* yfinance

## Tutorial and Examples

`PackCorrelation` uses a somewhat idiosyncratic data structure that makes it easy and quick to calculate correlations for each day for a stock to the alpha. The format uses `data` as a dictionary of dataframes with the stock tickers as keys. The `ticker_dates` dictionary is essentially a list of indices that point to locations in a ticker’s dataframe that correspond to the open and closing rows for each day. This allows for rapid lookups of individual price data or slicing of days in the case of Pack Correlation. The speed improvement comes at a cost of only the minor space requirement of `ticker_dates`. More details can be found in the complementary repository [here](https://github.com/leomcg108/FinData-Extraction). The `data` & `ticker_dates` data structure can be quickly implemented given csv files for each stock. The `FinDataExtract` class contains the `pop_data_dict` method which will generate the relevant dictionary and `pop_ticker_dates` will produce the `ticker_dates` structure.

```python
from findata_extraction import FinDataExtract

# Create instance of class
>>> fde = FinDataExtract()

# Define path at which to download data and/or from which to generate data and ticker_dates dictionaries
>>> fde.set_file_path(".//Watchlist//Test")

# create a list of tickers from a given csv file e.g. members of S&P 500 index
>>> fde.pop_watchlist(watchlist_path=".//Watchlist//sp500.csv")

# Download 4 weeks of 1m intraday data for each ticker from Yahoo Finance using the yfinance library 
>>> fde.download_ticker_data()

# Create a dictionary of dataframes as values and the stock ticker as corresponding key
>>> fde.pop_data_dict()
>>> data = fde.data

# Create a dictionary of lists of dates and open and close indices as values and stock ticker as corresponding key
>>> fde.pop_ticker_dates()
>>> ticker_dates = fde.ticker_dates

# Now we can easily access specific days of data for each ticker
# Access the opening info for most recent day for SPY
>>> open_index = ticker_dates["SPY"][-1][3]
>>> data["SPY"].loc[open_index]

Datetime     2022-04-01 09:30:00
Open                  453.309998
High                  453.339996
Low                   452.799988
Close                 453.070007
Adj Close             453.070007
Volume                 2290723.0
Name: 188977, dtype: object

# and the closing info for the most recent day
>>> close_index = ticker_dates["SPY"][-1][4]
>>> data["SPY"].loc[close_index]

Datetime     2022-04-01 15:59:00
Open                  453.140015
High                  453.170013
Low                   452.779999
Close                 452.890015
Adj Close             452.890015
Volume                 3033426.0
Name: 189366, dtype: object
```

Now that we have the correct data structures we can easily calculate the pack correlation for each day under consideration.

```python
# Create instance of the PackCorrelation class passing previously defined data and ticker_dates
>>> pack = PackCorrelation(data, ticker_dates)

# Define alpha against which correlations for all other members of pack will be calculated
>>> pack.define_alpha("SPY")

>>> print(pack)
Contains a dictionary of 560 dataframes with 252 days and alpha as SPY

# Run correlation calculation and plot result
>>> pack.find_pack_correlation(plot_av=True)
```
![image](https://user-images.githubusercontent.com/102587512/161964221-5b996a92-9f48-47ff-a7b4-3a58b14ce14f.png)

Here we can see the overall pack correlation as a function of time. The `find_pack_correlation` method will create a dataframe `pack.corr_date` containing the average, median and standard deviation of the entire pack correlation, the directional pack correlation (average pack correlation modified by alpha gain or loss) plus the names and correlations of the pack members (Beta, Epsilon, Sigma, Omega) for each day under consideration.

We can also easily pull out a single day and look at it in more depth. 

```python
>>> pack.corr_date.iloc[10]
Day             [4, 22, 2021]
Av Corr              0.626548
Dir Corr            -0.626548
Median Corr          0.749294
Stdev Corr           0.335698
Alpha Gain           0.991702
Beta                     NXPI
Beta Corr            0.979383
Epsilon                   TGT
Epsilon Corr         0.749294
Sigma                     CVS
Sigma Corr          -0.000276
Omega                    UVXY
Omega Corr          -0.976964
Name: 10, dtype: object

# can plot each of the pack members for a given date with prices normalized so as to be easily #compared
>>> pack.plot_day_corr(date="2022-02-25", plot_alpha=True, plot_beta=True, plot_omega=True)

Selected day: [2, 25, 2022]
Average day correlation: 0.77

Alpha: SPY
Beta: DHR (0.98)
Omega: UVXY (-0.96)
```
![image](https://user-images.githubusercontent.com/102587512/161966064-0edce062-7cf8-42c3-9059-272482a11548.png)

From this plot it’s clear to see that the Beta is following the Alpha closely and the Omega is anti-correlated with the two as is expected by definition. 

```python
# plot a histogram to show the distribution of correlations for a given day
>>> pack.plot_hist_corr("2022-04-01", bins=50)

Selected day: (2022, 4, 1)

Mean: 0.41
Median: 0.47
Mode: 0.56
```
![image](https://user-images.githubusercontent.com/102587512/161966250-0e64aff5-4e6a-4dd6-8cd5-edb4760efb24.png)

```python

# plot high and low correlated days together
>>> pack.plot_hist_corr("2022-02-25", bins=50, alpha=0.5)
>>> pack.plot_hist_corr("2021-11-02", bins=50, alpha=0.5)

Selected day: (2022, 2, 25)

Mean: 0.77
Median: 0.88
Mode: 0.91

Selected day: (2021, 11, 2)

Mean: 0.07
Median: 0.08
Mode: 0.46
```
![image](https://user-images.githubusercontent.com/102587512/161966364-a62e949a-4167-44cf-8af3-efdb56643832.png)

```python

# create and plot a heatmap to show all correlation distributions as a function of time
>>> pack.plot_heatmap(bins=100)
```
![image](https://user-images.githubusercontent.com/102587512/161966419-fc971b02-c9bc-46d1-8311-05074b7a5e42.png)

```python
# create a new dataframe for a ticker for the given date range
>>> new_data = pack.slice_data("DHR", "2022-02-22", "2022-02-28")
Data slice for DHR

# similarly, plot the “Close” data for a ticker for a given date range
>>> pack.plot_data("TSLA", start_time="2022-03-01", end_time="2022-03-31", plot_series="Close")
Data slice for TSLA
Close-data plotted for TSLA
```
![image](https://user-images.githubusercontent.com/102587512/161966478-c50c896b-58dc-4224-acd8-8fbd666c16f7.png)

## Comparison to simple correlations
It is reasonable to ask why we would use this pack correlation method over simply finding correlations between all the stocks of interest. This would give a more complete picture but would also include many spurious correlation we may not be interested; we want to know how the pack relates specifically to the alpha. Furthermore a simple calculation shows us that this complete approach may take significantly longer to achieve.

Take the above example, 560 stocks for the previous year of 252 trading days. In total there will be n(n-1)/2 correlation calculations between the stocks for any given day. For large enough numbers we can approximate this to n<sup>2</sup>  leading to, in the Big Oh notation, O(n<sup>2</sup>). Whereas for the pack correlation method we only have to make 559 correlation calculation of stocks to the alpha or O(n) in time complexity. In our above example this means approximately 280 times less work. This is one reason why this pack correlation method is clearly advantageous.

## Future Work
* For a given date, find correlations as a function of time
* Allow selection of different types of correlation methods
* Add weighting to stock correlations in pack when computing average correlation for a given day e.g. give AAPL a bigger contribution than A. Currently all stocks have equal weight
* Allow for pack correlation calculation for different granularity of data e.g. 5min, 1 hour, 1 day candles
* Add error estimation
* Provide features for cryptocurrencies

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

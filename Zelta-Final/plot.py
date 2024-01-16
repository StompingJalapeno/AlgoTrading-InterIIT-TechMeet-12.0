from copy import deepcopy

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio
from meta.config import TRAIN_START_DATE
# from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from pyfolio import timeseries

from meta import config


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def backtest_plot(
    account_value,
    baseline_start=config.TRADE_START_DATE,
    baseline_end=config.TRADE_END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns,
            benchmark_rets=baseline_returns,
            set_context=False,
        )


def get_baseline(ticker, start, end):
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def trx_plot(df_trade, df_actions, ticker_list):
    df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions["date"]
    df_trx.index.name = ""

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: x > 0)
        selling_signal = df_trx_temp_sign.apply(lambda x: x < 0)

        tic_plot = df_trade[
            (df_trade["tic"] == df_trx_temp.name)
            & (df_trade["date"].isin(df_trx.index))
        ]["close"]
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color="g", lw=2.0)
        plt.plot(
            tic_plot,
            "^",
            markersize=10,
            color="m",
            label="buying signal",
            markevery=buying_signal,
        )
        plt.plot(
            tic_plot,
            "v",
            markersize=10,
            color="k",
            label="selling signal",
            markevery=selling_signal,
        )
        plt.title(
            f"{df_trx_temp.name} Num Transactions: {len(buying_signal[buying_signal == True]) + len(selling_signal[selling_signal == True])}"
        )
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha="right")
        plt.show()



# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import matplotlib.dates as mdates
# %matplotlib inline
# #calculate agent returns
# account_value_erl = np.array(account_value_erl)
# agent_returns = account_value_erl/account_value_erl[0]
# #calculate buy-and-hold btc returns
# price_array = np.load('./price_array.npy')
# btc_prices = price_array[:,0]
# buy_hold_btc_returns = btc_prices/btc_prices[0]
# #calculate equal weight portfolio returns
# price_array = np.load('./price_array.npy')
# initial_prices = price_array[0,:]
# equal_weight = np.array([1e5/initial_prices[i] for i in range(len(TICKER_LIST))])
# equal_weight_values = []
# for i in range(0, price_array.shape[0]):
#     equal_weight_values.append(np.sum(equal_weight * price_array[i]))
# equal_weight_values = np.array(equal_weight_values)
# equal_returns = equal_weight_values/equal_weight_values[0]
# #plot 
# plt.figure(dpi=200)
# plt.grid()
# plt.grid(which='minor', axis='y')
# plt.title('Cryptocurrency Trading ', fontsize=20)
# plt.plot(agent_returns, label='ElegantRL Agent', color = 'red')
# plt.plot(buy_hold_btc_returns, label='Buy-and-Hold BTC', color='blue')
# plt.plot(equal_returns, label='Equal Weight Portfolio', color='green')
# plt.ylabel('Return', fontsize=16)
# plt.xlabel('Times (5min)', fontsize=16)
# plt.xticks(size=14)
# plt.yticks(size=14)
# '''ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(210))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(21))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
# ax.xaxis.set_major_formatter(ticker.FixedFormatter([]))'''
# plt.legend(fontsize=10.5)
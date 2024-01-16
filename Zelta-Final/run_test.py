from train import train
from test import test
from crypto_env_module import CryptoEnv
from plot import get_daily_return, backtest_stats
import numpy as np
from meta.data_processors.yahoofinance import Yahoofinance
import pickle
import pandas as pd
TICKER_LIST = ['BTCUSDT']
# env = CryptoEnv
TRAIN_START_DATE = '2017-07-01'
TRAIN_END_DATE = '2022-12-31'

# TEST_START_DATE = '2018-01-01'
# TEST_END_DATE = '2022-12-31'

TEST_START_DATE = '2017-01-01'# you can keep the testing dates as per wish
TEST_END_DATE = '2023-12-15'

# ultimate training period
# TEST_START_DATE = '2021-01-01' 
# TEST_END_DATE = '2022-04-01'

# INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet
INDICATORS = [
    "adx",
    "adxr",
    "aroon",
    "atr",
    "cci",
    "open",
    "high",
    "low",
    "volume",
    "macd",
]


TRAIN_FLAG=False
acc = test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = TICKER_LIST, 
                        test = True,
                        data_source = 'binance',
                        time_interval= '1d', 
                        technical_indicator_list= INDICATORS,
                        drl_lib='stable_baselines3', 
                        env=CryptoEnv, 
                        model_name='sac', 
                        current_working_dir='./models', 
                        net_dimension = 1024, 
                        if_vix=False
                        )





# # # plotting
df = pd.DataFrame(acc,columns=['account_value'])
df2 = pd.read_csv('data/dataset.csv')
df2['time'] = pd.to_datetime(df2['time']) 
mask = (df2['time'] > TEST_START_DATE) & (df2['time'] <= TEST_END_DATE)
df3 = df2.loc[mask]
df3 = df3.reset_index()
df3['account_value'] = pd.DataFrame(acc)
df3.rename(columns={'time': 'date'}, inplace=True)

backtest_result = backtest_stats(df3)
print(backtest_result)

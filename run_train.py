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
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2022-12-31'



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



TRAIN_FLAG = True
   
train(start_date=TRAIN_START_DATE,
      end_date=TRAIN_END_DATE,
      ticker_list=TICKER_LIST,
      data_source='binance',
      time_interval='1d',
      technical_indicator_list=INDICATORS,
      drl_lib='stable_baselines3',
      env=CryptoEnv,
      model_name='ppo',
      current_working_dir='./models',
      total_timesteps = 1e5, # The total number of samples (env steps) to train on
      break_step=+np.inf,
      if_vix=False,
      progress_bar=True
      )




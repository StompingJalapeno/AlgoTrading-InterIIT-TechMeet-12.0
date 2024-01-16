import numpy as np
import pandas as pd
import math
import gym
from gym import spaces
class CryptoEnv(gym.Env):
    signal_array = []
    data = {'Index': [], 'signals': [], 'trade_val_BTC':[], 'trade_value_USD': [], 'Cash':[], 'BTC':[], 'total_assets':[], 'Commission_fees':[]}
    signal_freq = {'0': 0, '1': 0, '-1': 0}
    total_commision = 0
    signal_chart = pd.DataFrame(data)
    FLAG = True
    delta = 0.1
    loss_trades = 0
    profit_trades = 0
    max_portfolio_value = 1e6
    min_portfolio_value = 1e6
    net_profit = 0

    ######################################
    csv_file_path = './data/dataset.csv'
    df1 = pd.read_csv(csv_file_path)
    df1.drop(columns= ['adjusted_close','tic'], inplace= True)
    
    df1['signals'] = ['']*df1['time'].size
    column_order = ['time', 'signals', 'open', 'high', 'low', 'close', 'volume']
    row_idx = 0
    df1 = df1[column_order]
    df1.rename(columns={'time': 'datetime'}, inplace=True)

    ######################################

    def save_df_train(self):
        file_path = 'results/summary_train.csv'
        self.signal_chart.to_csv(file_path, index=False)

    def save_df_test(self):
        file_path = 'results/summary_test.csv'
        self.signal_chart.to_csv(file_path, index=False)

    def save_df_test_logs(self):
        file_path = 'results/test_logs.csv'
        self.df1.to_csv(file_path, index=False)

    def set_flag(self, new_value):
        self.FLAG = new_value
    
    def __init__(self, config, lookback=1, initial_capital=1e6,
                 buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.95):
        self.stocks = 100
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = config['price_array']
        self.tech_array = config['tech_array']
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback

        # reset
        self.time = lookback-1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

        '''env information'''
        self.env_name = 'MulticryptoEnv'
        self.state_dim = 1 + (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback
        self.state_space = 1 + (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.action_dim = self.price_array.shape[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self.if_discrete = False
        self.target_return = 10

    def reset(self) -> np.ndarray:
        self.time = self.lookback-1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        state = self.get_state()
        return state

    def step(self, actions) -> (np.ndarray, float, bool, None):

        self.time += 1

        rl_agent_flag = True
        hold_flag = True

        # if self.time > 1:
        #     pct_price_change = ((self.price_array[self.time] - self.price_array[self.time - 1])/self.price_array[self.time - 1])*100
        #     if pct_price_change <= -2.5 and self.stocks != 0:
        #         sell_num_shares = self.stocks[-1]
        #         self.cash += sell_num_shares*self.price_array[-1] * 0.999
        #         rl_agent_flag = False
        #         self.loss_trades+=1
        #         self.signal_freq['-1'] += 1
        #         self.signal_array.append(-1)
        #         # input()
        #     elif pct_price_change >= 5 and self.stocks != 0:
        #         sell_num_shares = self.stocks[-1]
        #         self.cash += sell_num_shares*self.price_array[-1] * 0.999
        #         rl_agent_flag = False
        #         self.profit_trades+=1
        #         self.signal_freq['-1'] += 1
        #         self.signal_array.append(-1)
        #         # input()
        #     else:
        #         self.signal_freq['0'] += 1
        
        try:
            price = self.price_array[self.time]
        except:
            print(f'type price array = {type(self.price_array)}')
            print(f'self time = {self.time}')
            print(f'price array size = {self.price_array.shape}')
            exit()


        if rl_agent_flag:
            for i in range(self.action_dim):
                norm_vector_i = self.action_norm_vector[i]
                actions[i] = actions[i] * norm_vector_i

            for index in np.where(actions < -1*self.delta)[0]:  # sell_index:
                if price[index] > 0 and self.stocks != 0:  # Sell only if the current asset is > 0
                    print("---------------------selling---------------------------------------")
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    value_of_trade = price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    self.cash += value_of_trade
                    if self.FLAG:
                        self.signal_array.append(-1)
                        # self.signal_chart.loc[len(self.signal_chart.index)] = [price[index], -1, sell_num_shares, value_of_trade, self.cash, self.stocks, self.total_asset]  
                        # new_row_data = {'Index': 1, 'signals': 'Buy', 'trade_val_BTC': 0.5, 'trade_value_USD': 500, 'Cash': 1000, 'BTC': 2, 'total_assets': 1500}
                        # self.signal_chart = self.signal_chart._append(new_row_data, ignore_index=True)
                        new_row_data = {'Index': price[index], 'signals': -1, 'trade_val_BTC': sell_num_shares, 'trade_value_USD': value_of_trade, 'Cash': self.cash, 'BTC': self.stocks[index], 'total_assets': self.total_asset, 'Commission_fees': 0.001*value_of_trade}
                        self.signal_chart = self.signal_chart._append(new_row_data, ignore_index=True)
                        self.df1['signals'][self.row_idx] = -1
                        self.row_idx += 1
                        self.signal_freq['-1'] += 1
                        self.total_commision += 0.001*value_of_trade
                        hold_flag = False
                        
                        


            for index in np.where(actions > self.delta)[0]:  # buy_index:
                if price[index] > 0 and self.cash > 0:  # Buy only if the price is > 0 (no missing data on this particular date)
                    print("---------------------buying---------------------------------------")
                    buy_num_shares = min(self.cash // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    value_of_trade = price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.cash -= value_of_trade
                    if self.FLAG:
                        self.signal_array.append(1)
                        # self.signal_chart.loc[len(self.signal_chart.index)] = [price[index], 1, buy_num_shares, value_of_trade, self.cash, self.stocks, self.total_asset]
                        new_row_data = {'Index': price[index], 'signals': 1, 'trade_val_BTC': buy_num_shares, 'trade_value_USD': value_of_trade, 'Cash': self.cash, 'BTC': self.stocks[index], 'total_assets': self.total_asset, 'Commission_fees': 0.001*value_of_trade}
                        self.signal_chart = self.signal_chart._append(new_row_data, ignore_index=True)
                        self.df1['signals'][self.row_idx] = 1
                        self.row_idx += 1
                        self.signal_freq['1'] += 1
                        self.total_commision += 0.001*value_of_trade
                        hold_flag = False

            for index in np.where((actions > -1*self.delta) & (actions < self.delta))[0]:
                if self.FLAG:
                    self.signal_array.append(0)
                    # self.signal_chart.loc[len(self.signal_chart.index)] = [price[index], 0, 0, 0,  self.cash, self.stocks, self.total_asset]
                    new_row_data = {'Index': price[index], 'signals': 0, 'trade_val_BTC': 0, 'trade_value_USD': 0, 'Cash': self.cash, 'BTC': self.stocks[index], 'total_assets': self.total_asset, 'Commission_fees': 0}
                    self.signal_chart = self.signal_chart._append(new_row_data, ignore_index=True)
                    self.df1['signals'][self.row_idx] = 0
                    self.row_idx += 1
                    self.signal_freq['0'] += 1
                # input()

            if hold_flag:
                self.signal_array.append(0)
                # self.signal_chart.loc[len(self.signal_chart.index)] = [price[index], 0, 0, 0,  self.cash, self.stocks, self.total_asset]
                new_row_data = {'Index': price[index], 'signals': 0, 'trade_val_BTC': 0, 'trade_value_USD': 0, 'Cash': self.cash, 'BTC': self.stocks[index], 'total_assets': self.total_asset, 'Commission_fees': 0}
                self.signal_chart = self.signal_chart._append(new_row_data, ignore_index=True)
                self.signal_freq['0'] += 1
                self.df1['signals'][self.row_idx] = 0
                self.row_idx += 1

        """update time"""
        done = self.time == self.max_step

        if self.total_asset > self.max_portfolio_value:
            self.max_portfolio_value = self.total_asset
        
        if self.total_asset < self.min_portfolio_value:
            self.min_portfolio_value = self.total_asset
        
        if True:
            # print(self.signal_array)
            # print(self.signal_chart)
            print("--------------------------------------------------------")
            print(f'self.cash = {float(self.cash)}')
            print(f'self.stocks = {float(self.stocks)}')
            print(f'self.total_asset = {float(self.total_asset)}')
            print("--------------------------------------------------------")
            print(f'net_profit = {float(self.total_asset - self.initial_cash)}')
            print(self.signal_freq)
            print(f'commission at 0.1%(already included in net profit calculated above) = {self.total_commision}')
            print(f'max_portfolio_value = {self.max_portfolio_value}')
            print(f'min_portfolio_value = {self.min_portfolio_value}')
           

        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash
        if done:
            # print(self.signal_array)
            reward = self.gamma_return
            print(f'reward = {float(reward)}')
            self.episode_return = self.total_asset / self.initial_cash
            print(f'self.episode_return = {self.episode_return}')
        return state, reward, done, {}

    def get_state(self):
        state = np.hstack((self.cash * 2 ** -18, self.stocks * 2 ** -3))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]
            normalized_tech_i = tech_i * 2 ** -15
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)
        return state

    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        print(price_0)
        for price in price_0:
            x = math.floor(math.log(price, 10))  # the order of magnitude
            action_norm_vector.append(1 / ((10) ** x))

        action_norm_vector = np.asarray(action_norm_vector) * 10000
        self.action_norm_vector = np.asarray(action_norm_vector)

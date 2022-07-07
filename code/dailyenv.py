from typing import List, Callable
import datetime as dt

import gym
from gym import spaces
import numpy as np
import pandas as pd

class DailyTradingEnv(gym.Env):

    def __init__(self, tickers: List[str], start_date: dt.datetime, end_date: dt.datetime, directions_src: str) -> None:
        super(DailyTradingEnv, self).__init__()
        self.tickers = tickers
        self.start = start_date
        self.end_date = end_date
        
        self.dates = pd.read_csv(f'../data/day_data/{tickers[0]} MK Equity.csv', parse_dates=True, index_col="Dates").loc[start_date:end_date].index.to_numpy()
        self.logs = {'actions': [], 'rewards': [], 'bankroll': []}

        self.last_prices = []
        self.directions = []
        self.dividends = []

        self.reward_range = (0, np.inf)
        self.action_space = spaces.MultiBinary(len(tickers) + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(tickers),))

        for ticker in tickers:
            day_data_df = pd.read_csv(f'../data/day_data/{ticker} MK Equity.csv', parse_dates=True, index_col="Dates").loc[start_date:end_date]
            self.last_prices.append(day_data_df["PX_LAST"].to_numpy())
            
            directions_df = pd.read_csv(f'../data/directions/{directions_src}/Directions {ticker}.csv', parse_dates=True, index_col="Dates").loc[start_date:end_date][1:]
            directions_np = (directions_df.drop(columns='AVG') == 1).sum(axis=1).to_numpy()/10
            self.directions.append(directions_np)
            
            dividend_df = pd.read_csv(f'../data/dividends/{ticker} dividend.csv', parse_dates=True, index_col="Date").loc[start_date:end_date]
            s = pd.Series(data=0, index=self.dates)
            for k, v in dividend_df['Dividends'].iteritems():
                s[k] = v
            self.dividends.append(s.to_numpy())
        
        self.last_prices = np.array(self.last_prices).T
        self.directions = np.array(self.directions).T
        self.dividends = np.array(self.dividends).T

        self.period_length = len(self.last_prices) - 2

        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        
    def step(self, action: np.ndarray):
        assert len(action) == len(self.tickers) + 1

        normalize = np.sum(action)
        if normalize > 0:
            weights = action / np.sum(action)
        else:
            weights = action
        
        curr_prices = self.last_prices[self.curr_index]
        next_prices = self.last_prices[self.curr_index + 1]
        next_day_div = self.dividends[self.curr_index + 1]

        shares = np.floor(self.current_balance * weights[1:] / curr_prices)
        profits = np.dot(next_prices - curr_prices, shares)
        div_received = np.dot(next_day_div, shares)

        reward = self._calc_reward(profits, div_received)

        self.current_balance += profits + div_received
        self.balance_record.append(self.current_balance)
        self.curr_index += 1

        self.last_reward = reward
    
        obs, rew, done, info = self._get_obs(), reward, self.curr_index == self.period_length, {}

        return obs, rew, done, info

    def reset(self):
        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        self.balance_record = [self.current_balance]
        return self._get_obs()

    def render(self, mode="human"):
        return super().render(mode)

    def _calc_reward(self, profit, div_received):
        reward = (profit + div_received) / self.current_balance
        if reward < 0:
            reward *= 1.2
        return reward

    def _get_obs(self):
        return self.directions[self.curr_index]

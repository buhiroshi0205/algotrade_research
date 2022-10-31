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
        # action space is equivalent to a boolean array. Index 0 is true if and only if we hold cash, and the Index i is true if and only if we buy the i-1'th stock today (and sell tomorrow)
        # Note that there aren't transaction costs yet. Money is divided evenly into each stock that we choose to buy.
        self.action_space = spaces.MultiDiscrete(tuple([3]*(len(tickers))))
        # observation space is a float array. Index i = the fraction of the ensemble models that predict the i'th stock to go up
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(tickers),2))

        # load data
        for ticker in tickers:
            # ground truth price data
            day_data_df = pd.read_csv(f'../data/day_data/{ticker} MK Equity.csv', parse_dates=True, index_col="Dates").loc[start_date:end_date]
            self.last_prices.append(day_data_df["PX_LAST"].to_numpy())
            
            # predicted directions from the ensemble
            directions_df = pd.read_csv(f'../data/directions/{directions_src}/Directions {ticker}.csv', parse_dates=True, index_col="Dates").loc[start_date:end_date][1:]
            directions_np = (directions_df.drop(columns='AVG') == 1).sum(axis=1).to_numpy()/10
            self.directions.append(directions_np)
            
            # dividends (currently mostly ignored since there is little dividend)            
            dividend_df = pd.read_csv(f'../data/dividends/{ticker} dividend.csv', parse_dates=True, index_col="Date").loc[start_date:end_date]
            s = pd.Series(data=0, index=self.dates)
            for k, v in dividend_df['Dividends'].items():
                s[k] = v
            self.dividends.append(s.to_numpy())
        
        self.last_prices = np.array(self.last_prices).T
        self.directions = np.array(self.directions).T
        self.dividends = np.array(self.dividends).T

        self.period_length = len(self.last_prices) - 2

        self.curr_portfolio = np.zeros(len(tickers))
        self.cost_basis = np.zeros(len(tickers))
        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        # Cost as fraction of transaction amount
        self.transaction_cost = 0.01
        
    def step(self, action: np.ndarray):
        assert len(action) == len(self.tickers)


        sell_mask = np.logical_and((action == 0), self.curr_portfolio > 0)
        #hold_mask = (action == 1)
        buy_mask = (action == 2)
        
        # basically given which stocks to buy today and sell next day, calculate profits and add to bankroll
        curr_prices = self.last_prices[self.curr_index]
        next_prices = self.last_prices[self.curr_index + 1]
        next_day_div = self.dividends[self.curr_index + 1]
        
        # First sell stocks
        profits = np.dot(curr_prices[sell_mask], self.curr_portfolio[sell_mask])
        profits -= np.sum(np.dot(curr_prices[sell_mask], self.curr_portfolio[sell_mask]))*self.transaction_cost
        
        self.current_balance += profits
        self.curr_portfolio[sell_mask] = 0
        self.cost_basis[sell_mask] = 0
        
        # Then buy stocks evenly
        weights = np.zeros(action.shape)
        weights[buy_mask] = action[buy_mask] / np.sum(action[buy_mask])

        new_shares = np.floor(self.current_balance * weights / (curr_prices*(1+self.transaction_cost)))
        profits -= np.sum(np.dot(curr_prices, new_shares))*(1+self.transaction_cost)
        self.current_balance -= np.sum(np.dot(curr_prices, new_shares))*(1+self.transaction_cost)
        
        self.cost_basis[buy_mask] = ((self.cost_basis[buy_mask]*self.curr_portfolio[buy_mask]) + (curr_prices[buy_mask]*new_shares[buy_mask]))/(self.curr_portfolio[buy_mask]+new_shares[buy_mask]+ (1e-8))
        self.curr_portfolio += new_shares
        
        div_received = np.dot(next_day_div, self.curr_portfolio)        

        reward = self._calc_reward(profits, div_received)

        self.current_balance += div_received
        self.balance_record.append(self.current_balance + np.dot(self.curr_portfolio, curr_prices))
        self.curr_index += 1

        self.last_reward = reward
    
        obs, rew, done, info = self._get_obs(), reward, self.current_balance <= 0 or self.curr_index == self.period_length, {}

        return obs, rew, done, info

    def reset(self):
        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        self.balance_record = [self.current_balance]
        self.curr_portfolio = np.zeros(self.curr_portfolio.shape)
        self.cost_basis = np.zeros(self.curr_portfolio.shape)
        return self._get_obs()

    def render(self, mode="human"):
        return super().render(mode)

    def _calc_reward(self, profit, div_received):
        # percentage reward so reward doesn't change depending on how much money we initially had
        reward = (profit + div_received) / self.current_balance
        # penalize losses (theoretically would hurt CAGR but improve sharpe ratio, as it leads to less risks and volatility)
        if reward < 0:
            reward *= 1.2
        return reward

    def _get_obs(self):
        #return self.directions[self.curr_index]
        return np.array([self.directions[self.curr_index], self.curr_portfolio]).T

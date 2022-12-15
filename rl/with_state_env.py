import gym
import numpy as np
import pandas as pd
import random

class StateEnv(gym.Env):
    
    def __init__(self, tickers, timeframes, starting_balance=1000000, tc=0.00):
        super(StateEnv, self).__init__()
        self.tickers = tickers
        self.timeframes = timeframes
        self.action_space = gym.spaces.MultiBinary(len(self.tickers))
        self.observation_space = gym.spaces.Box(low=0,high=1, shape=(2,len(self.tickers)))
        self.starting_balance = starting_balance
        self.curr_portfolio = np.zeros(len(tickers))
        self.tc=tc
        self.reset()
        
    def step(self, action: np.ndarray):
        assert len(action) == len(self.tickers)
        weights = action
        if action.sum() > 0:
            weights = action / action.sum()
        
        curr_prices = self.prices[self.curr_idx]
        next_prices = self.prices[self.curr_idx + 1]
        next_day_div = self.dividends[self.curr_idx + 1]
        
        shares = np.floor(self.current_balance * weights / curr_prices)
        profits = np.dot(next_prices - curr_prices, shares)-np.sum(np.dot(curr_prices,
                         np.abs(shares - self.curr_portfolio)))*self.tc
        div_received = np.dot(next_day_div, shares)
        reward = self._calc_reward(profits, div_received)

        self.current_balance += profits+div_received
        self.history.append(self.current_balance)
        self.curr_idx += 1

        done=False
        if self.curr_idx == self.num_timesteps or self.current_balance <= 0:
            done = True
        return self._get_obs(), reward, done, {}

    
    def reset(self):
        self.curr_idx = 0
        self.current_balance = self.starting_balance
        self.history = [self.current_balance]
        timeframe = random.choice(self.timeframes)
        self.directions = []
        self.prices = []
        self.dividends=[]
        for ticker in self.tickers:
            day_data_df = pd.read_csv(f'../data/Day Data with Volatility/{ticker} MK Equity.csv',
                                      parse_dates=True, index_col="Dates").loc[timeframe[0][0]:timeframe[0][1]]
            self.prices.append(day_data_df["PX_LAST"].to_numpy())
            
            directions_df = pd.read_csv(f'../data/directions/{timeframe[1]}/Directions {ticker}.csv', 
                                        parse_dates=True, index_col="Dates").loc[timeframe[0][0]:timeframe[0][1]][1:]
            directions_np = (directions_df.drop(columns='AVG') == 1).sum(axis=1).to_numpy()/10
            self.directions.append(directions_np)
            dividend_df = pd.read_csv(f'../data/dividends/{ticker} dividend.csv',
                                      parse_dates=True, index_col="Date").loc[timeframe[0][0]:timeframe[0][1]]
            s = pd.Series(data=0, index=day_data_df.index)
            for k, v in dividend_df['Dividends'].items():
                s[k] = v
            self.dividends.append(s.to_numpy())
        self.directions = np.array(self.directions).T
        self.prices = np.array(self.prices).T
        self.dividends = np.array(self.dividends).T
        self.num_timesteps = len(self.prices)-2
        return self._get_obs()

    
    def _calc_reward(self, profit, div_received):
        reward = (profit + div_received) / self.current_balance
        if reward < 0:
            reward *= 1.2
        return reward
    

    def _get_obs(self):
        return np.vstack((self.directions[self.curr_idx],self.curr_portfolio*self.prices[self.curr_idx]/self.current_balance))

from typing import List, Callable
import datetime as dt
import os
from statistics import stdev

import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import quantstats as qs
from tqdm import tqdm

class DailyTradingEnv(gym.Env):

    def __init__(self, tickers: List, start_date: dt.datetime, end_date: dt.datetime, logname: str = None, redistribution: float = 0) -> None:
        super(DailyTradingEnv, self).__init__()
        self.tickers = tickers
        self.start = start_date
        self.end_date = end_date
        self.pretrain = False
        
        self.dates = pd.read_csv(os.path.join("../data", "day_data", f"{tickers[0]} MK Equity.csv"), parse_dates=True, index_col="Dates").loc[start_date:end_date].index.to_numpy()
        self.logs = {'actions': [], 'rewards': [], 'bankroll': []}

        self.last_prices = []
        self.directions = []
        self.dividends = []

        self.reward_range = (0, np.inf)
        self.action_space = spaces.MultiBinary(len(tickers) + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(tickers),))

        for ticker in tqdm(tickers):
            day_data_df = pd.read_csv(os.path.join("../data", "day_data", f"{ticker} MK Equity.csv"), parse_dates=True, index_col="Dates").loc[start_date:end_date]
            self.last_prices.append(day_data_df["PX_LAST"].to_numpy())
            
            directions_df = pd.read_csv(os.path.join("../data/directions/2010split", f"Directions {ticker}.csv"), parse_dates=True, index_col="Dates").loc[start_date:end_date][1:]
            if 'AVG' in directions_df:
                directions_df = directions_df.drop(columns='AVG')
            directions_np = (directions_df==1).sum(axis=1).to_numpy()/10
            if redistribution > 0:
                random_loc = np.random.rand(*directions_np.shape) < redistribution
                random_dir = np.random.randint(0,11,size=directions_np.shape)/10
                self.directions.append(np.where(random_loc, random_dir, directions_np))
            else:
                self.directions.append(directions_np)
            
            dividend_df = pd.read_csv(os.path.join("../data", "dividends", f"{ticker} dividend.csv"), parse_dates=True, index_col="Date").loc[start_date:end_date]
            s = pd.Series(data=0, index=self.dates)
            for k, v in dividend_df['Dividends'].iteritems():
                s[k] = v
            self.dividends.append(s.to_numpy())
            
            assert (day_data_df.index[1:] == directions_df.index).all()
            assert (day_data_df.index == s.index).all()
        
        self.last_prices = np.array(self.last_prices).T
        self.directions = np.array(self.directions).T
        self.dividends = np.array(self.dividends).T

        self.period_length = len(self.last_prices) - 2

        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        
        self.tf_logger = TensorBoardLogger('tensorboard_log', logname) if logname is not None else None
        
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

        if self.pretrain:
            target = np.zeros(len(self.tickers)+1)
            for i in range(len(self.tickers)):
                if self.last_obs[i] * 10 >= 10:
                    target[i+1] = 1
            #target = np.concatenate(([0.], self.desired_action))
            if np.sum(target) > 0:
                reward = 0.1 if np.all(target == action) else 0
            else:
                reward = 0.01 if np.all(target == action) else 0
        else:
            reward = self._calc_reward(profits, div_received)

        self.current_balance += profits + div_received
        self.curr_index += 1

        self.last_reward = reward
    
        obs, rew, done, info = self._get_obs(), reward, self.curr_index == self.period_length, {}
        if self.pretrain:
            self.last_obs = obs
       
        self.current_log['actions'].append(action)
        self.current_log['rewards'].append(rew)
        self.current_log['bankroll'].append(self.current_balance)
        if done:
            for k in self.logs.keys():
                self.logs[k].append(np.array(self.current_log[k]))
            if self.tf_logger is not None:
                self.tf_logger.log('custom/reward', np.mean(self.logs['rewards'][-1]))
                br_series = self.get_series('bankroll', -1)
                self.tf_logger.log('custom/CAGR', qs.stats.cagr(br_series))
                self.tf_logger.log('custom/sharpe', qs.stats.sharpe(br_series))
        return obs, rew, done, info

    def reset(self):
        self.curr_index = 0
        self.current_balance = 100000
        self.last_reward = 0
        self.current_log = {'actions': [], 'rewards': [], 'bankroll': []}
        obs = self._get_obs()
        if self.pretrain:
            self.last_obs = obs
        
        self.current_log['bankroll'].append(self.current_balance)
        return obs

    def render(self, mode="human"):
        return super().render(mode)

    def _calc_reward(self, profit, div_received):
        reward = (profit + div_received) / self.current_balance
        if reward < 0:
            reward *= 1.2
        return reward

    def _get_obs(self):
        return self.directions[self.curr_index]
    
    def clear_logs(self):
        self.logs = {'actions': [], 'rewards': [], 'bankroll': []}
        self.current_log = {'actions': [], 'rewards': [], 'bankroll': []}
        
    def save_logs(self, path):
        save_dict = {}
        save_dict['dates'] = self.dates
        save_dict['packedactions'] = np.packbits(np.array(self.logs['actions'], dtype=bool), axis=1)
        save_dict['rewards'] = np.array(self.logs['rewards'])
        save_dict['bankroll'] = np.array(self.logs['bankroll'])
        np.savez(path, **save_dict)
    
    def load_logs(self, path):
        load_dict = np.load(path)
        self.dates = load_dict['dates']
        self.logs['actions'] = list(np.unpackbits(load_dict['packedactions'], axis=1)[:,:len(self.dates),:])
        self.logs['rewards'] = list(load_dict['rewards'])
        self.logs['bankroll'] = list(load_dict['bankroll'])
    
    def get_logs_len(self):
        return len(self.logs['bankroll'])
    
    def get_series(self, name, i):
        return pd.Series(self.logs[name][i], index=self.dates[len(self.dates)-len(self.logs[name][i]):])
            
    def plot_change(self, metric, ax=None, xlabel='episodes'):
        settitle = plt.title if ax is None else ax.set_title
        setxlabel = plt.xlabel if ax is None else ax.set_xlabel
        drawplot = plt.plot if ax is None else ax.plot
        
        #settitle(metric if metric == 'rewards' else metric.__name__)
        #setxlabel(xlabel)
        drawplot(np.mean(self.logs['rewards'], axis=1) if metric == 'rewards' else [metric(self.get_series('bankroll',i)) for i in range(self.get_logs_len())])
    
    def report(self, report='basic', i=-1, **kwargs):
        eval(f'qs.reports.{report}')(self.get_series('bankroll',i), **kwargs)

        

        
class TensorBoardLogger:
    
    def __init__(self, dir, name):
        self.writer = SummaryWriter(os.path.join(dir, name))
        self.step = 0
    
    def inc_step(self):
        self.step += 1
    
    def log(self, key, value):
        self.writer.add_scalar(key, value, self.step)
    
    

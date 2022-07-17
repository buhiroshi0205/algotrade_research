import datetime as dt

import numpy as np
import pandas as pd
import quantstats as qs

from dailyenv import DailyTradingEnv


all_stocks = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]


def threshold(obs, n):
    action = np.zeros(len(obs)+1)
    for i in range(len(obs)):
        if obs[i] * 10 >= n:
            action[i+1] = 1
    if np.sum(action) == 0:
        action[0] = 1
    return action


for n in range(11):
    env = DailyTradingEnv(all_stocks[:5], dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1), 'olivier')
    obs = env.reset()
    done = False
    while not done:
        action = threshold(obs, n=n)
        obs, reward, done, info = env.step(action)
    series = pd.Series(env.balance_record, index=env.dates[:-1])
    print(f'{n}: {qs.stats.sharpe(series):.2f} {qs.stats.cagr(series)*100:.2f}%')
import os
import datetime as dt
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
pd.options.mode.chained_assignment = None

LABEL = "PX_LAST"
FEATURES = [LABEL, 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME',
            'EQY_WEIGHTED_AVG_PX', 'MOV_AVG_5D', 'MOV_AVG_10D',
            'MOV_AVG_20D','MOV_AVG_30D', 'MOV_AVG_40D', 'MOV_AVG_50D',
            'MOV_AVG_60D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'MOV_AVG_180D',
            'MOV_AVG_200D','REALIZED_VOL_3D', 'REALIZED_VOL_5D',
            'REALIZED_VOL_10D', 'REALIZED_VOL_20D', 'REALIZED_VOL_50D',
            'RSI_3D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'DOW_0', 'DOW_1',
            'DOW_2', 'DOW_3', 'DOW_4', 'Q_1', 'Q_2', 'Q_3', 'Q_4']

RSI_RANGE_START = 22
RSI_RANGE_END = 26

class DailyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, look_backward: int, predict_range=1, dividend_df=None) -> None:
        self.df = df.dropna()

        # Dates preprocessing
        self.df["Dates"] = pd.to_datetime(self.df["Dates"])
        self.df["DAYOFWEEK"] = self.df["Dates"].dt.dayofweek
        self.df["QUARTER"] = self.df["Dates"].dt.quarter
        self.df = pd.get_dummies(self.df, prefix=["DOW", "Q"], columns=["DAYOFWEEK", "QUARTER"])

        # Dividends
        self.df["PX_LAST_ADJUSTED"] = self.df["PX_LAST"]
        self.dividend_df = dividend_df
        if dividend_df is not None:
            dividend_df["Date"] = pd.to_datetime(dividend_df["Date"])
            for i in dividend_df.index:
                date = dividend_df.loc[i, "Date"]
                amount = dividend_df.loc[i, "Dividends"]
                self.df["PX_LAST_ADJUSTED"] = self.df.apply(lambda x: x.PX_LAST_ADJUSTED - amount if x.Dates < date else x.PX_LAST_ADJUSTED, axis=1)

        self.look_backward = look_backward
        self.predict_range = predict_range
        self.use_index = self.df.index[look_backward:]

        self.tensorx = torch.tensor(self.df[FEATURES].values, dtype=torch.float)
        self.tensory = torch.tensor(self.df[LABEL].values, dtype=torch.float)
        self.adjusted = self.df["PX_LAST_ADJUSTED"].values
    
    def __len__(self) -> int:
        return len(self.use_index)

    def __getitem__(self, index: int, return_scale: bool=False) -> Tuple[Tensor, Tensor, Tensor]:
        now = index + self.look_backward
        x = self.tensorx[now-self.look_backward:now].clone()
        if self.predict_range == 1:
            y = self.tensory[now].clone()
        else:
            y = self.tensory[now-self.predict_range+1:now+1].clone()

        # Standardization
        x_std, x_mean = torch.std_mean(x[:, :RSI_RANGE_START], dim=0, unbiased=True)
        x[:, :RSI_RANGE_START] = (x[:, :RSI_RANGE_START] - x_mean) / x_std
        y = (y - x_mean[0]) / x_std[0]
        x = torch.nan_to_num(x, 0)
        y = torch.nan_to_num(y, 0)
        
        # Transform RSI to 0-1 range
        if RSI_RANGE_END > RSI_RANGE_START:
            x[:, RSI_RANGE_START:RSI_RANGE_END] /= 100

        move = self.adjusted[index+self.look_backward] / self.adjusted[index+self.look_backward-1] - 1

        if return_scale:
            return x, y, move, x_mean[0], x_std[0]
        return x, y, move

def get_daily_dataset(df: pd.DataFrame, look_backward: int, val_start: dt.datetime,
                      test_start: dt.datetime, predict_range=1, div_df=None) -> Tuple[DailyDataset, DailyDataset, DailyDataset]:
    all_dates = pd.to_datetime(df["Dates"])

    first_val = df.index[all_dates >= val_start][0]
    first_test = df.index[all_dates >= test_start][0]
    train_df = df.loc[:first_val]
    val_df = df.loc[first_val - look_backward + 1: first_test]
    test_df = df.loc[first_test - look_backward + 1:]
    return DailyDataset(train_df, look_backward, predict_range, div_df), DailyDataset(val_df, look_backward, predict_range, div_df), DailyDataset(test_df, look_backward, predict_range, div_df)

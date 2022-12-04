import os
import datetime as dt
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
pd.options.mode.chained_assignment = None

LABEL = "PX_OPEN"
LABEL_ADJUSTED = LABEL + '_ADJUSTED'
FEATURES = [
    'PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME',
    'EQY_WEIGHTED_AVG_PX', 'MOV_AVG_5D', 'MOV_AVG_10D',
    'MOV_AVG_20D','MOV_AVG_30D', 'MOV_AVG_40D', 'MOV_AVG_50D',
    'MOV_AVG_60D', 'MOV_AVG_100D', 'MOV_AVG_120D', 'MOV_AVG_180D',
    'MOV_AVG_200D','REALIZED_VOL_3D', 'REALIZED_VOL_5D',
    'REALIZED_VOL_10D', 'REALIZED_VOL_20D', 'REALIZED_VOL_50D',
    'RSI_3D', 'RSI_9D', 'RSI_14D', 'RSI_30D', 'DOW_0', 'DOW_1',
    'DOW_2', 'DOW_3', 'DOW_4', 'Q_1', 'Q_2', 'Q_3', 'Q_4', 
    'SP_OPEN', 'SP_HIGH', 'SP_LOW', 'SP_CLOSE'
]

RSI_RANGE_START = 22
RSI_RANGE_END = 26


"""
A pytorch dataset class to streamline the training process. Data is structured so model uses N days of data to predict M days of price,
where the N'th day of input data is the same day as the M-1'th day of output data. We end up predicting only one new day of prices, while
the other days are there so certain transformers perform better.

   N days
|----------|
        |---|
        M days

X data is two-dimensional: one dimension is the days we look backward (size N), and the other dimension is all the features in `FEATURES`
Y data is one-dimensional: the prices to predict.

Both X and Y data are standardized, so we'll need to un-standardize the data for accurate pricing predictions.

:param look_backward: = N
:param predict_range: = M
"""
class DailyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, look_backward: int, predict_range=1, dividend_df=None) -> None:
        # Load S&P data
        df_sp = pd.read_csv("../data/index_data/SP500.csv")
        df_sp['Dates']= pd.to_datetime(df_sp['Dates'])
        
        # Merge with given df
        df["Dates"] = pd.to_datetime(df["Dates"])
        # Trading days in the two markets may not match. Hence, left join
        df = df.merge(df_sp, on=['Dates'], how = "left")
        
        # Makes the left join above pointless. Consider interpolating S&P data to prevent data loss instead?
        self.df = df.dropna()

        # Dates preprocessing
        self.df["DAYOFWEEK"] = self.df["Dates"].dt.dayofweek
        self.df["QUARTER"] = self.df["Dates"].dt.quarter
        self.df = pd.get_dummies(self.df, prefix=["DOW", "Q"], columns=["DAYOFWEEK", "QUARTER"])

        # Dividends, probably not that much useful
        self.df[LABEL_ADJUSTED] = self.df[LABEL]
        self.dividend_df = dividend_df
        if dividend_df is not None:
            dividend_df["Date"] = pd.to_datetime(dividend_df["Date"])
            for i in dividend_df.index:
                date = dividend_df.loc[i, "Date"]
                amount = dividend_df.loc[i, "Dividends"]
                self.df[LABEL_ADJUSTED] = self.df.apply(lambda x: x[LABEL_ADJUSTED] - amount if x.Dates < date else x[LABEL_ADJUSTED], axis=1)

        self.look_backward = look_backward
        self.predict_range = predict_range
        self.use_index = self.df.index[look_backward:]

        # store in tensor for fast retrieval
        self.tensorx = torch.tensor(self.df[FEATURES].values, dtype=torch.float)
        self.tensory = torch.tensor(self.df[LABEL].values, dtype=torch.float)
        self.adjusted = self.df[LABEL_ADJUSTED].values
    
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


# Utility for obtaining train, eval, and test datasets together given two splits (val_start, test_start)
def get_daily_dataset(df: pd.DataFrame, look_backward: int, val_start: dt.datetime,
                      test_start: dt.datetime, predict_range=1, div_df=None) -> Tuple[DailyDataset, DailyDataset, DailyDataset]:
    all_dates = pd.to_datetime(df["Dates"])

    first_val = df.index[all_dates >= val_start][0]
    first_test = df.index[all_dates >= test_start][0]
    train_df = df.loc[:first_val]
    val_df = df.loc[first_val - look_backward + 1: first_test]
    test_df = df.loc[first_test - look_backward + 1:]
    return DailyDataset(train_df, look_backward, predict_range, div_df), DailyDataset(val_df, look_backward, predict_range, div_df), DailyDataset(test_df, look_backward, predict_range, div_df)

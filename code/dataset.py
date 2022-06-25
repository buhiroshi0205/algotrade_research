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
    def __init__(self, df: pd.DataFrame, look_backward: int, dividend_df=None) -> None:
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
        self.use_index = self.df.index[look_backward:]
    
    def __len__(self) -> int:
        return len(self.use_index)

    def __getitem__(self, index: int, return_scale: bool=False) -> Tuple[Tensor, Tensor, Tensor]:
        i = self.use_index[index]
        x = torch.tensor(self.df.loc[(i - self.look_backward):(i-1)][FEATURES].values)
        y = torch.tensor(self.df.loc[i][LABEL])

        # Standardization
        x_std, x_mean = torch.std_mean(x[:, :RSI_RANGE_START], dim=0, unbiased=True)
        x[:, :RSI_RANGE_START] = (x[:, :RSI_RANGE_START] - x_mean) / x_std
        y = (y - x_mean[0]) / x_std[0]
        x = torch.nan_to_num(x, 0)
        y = torch.nan_to_num(y, 0)
        
        # Transform RSI to 0-1 range
        if RSI_RANGE_END > RSI_RANGE_START:
            x[:, RSI_RANGE_START:RSI_RANGE_END] /= 100

        move = torch.tensor(self.df.loc[i]["PX_LAST_ADJUSTED"] / self.df.loc[i-1]["PX_LAST_ADJUSTED"] - 1)

        if return_scale:
            return x.double(), y.double(), move.double(), x_mean[0], x_std[0]
        return x.double(), y.double(), move.double()

class DatasetFFD(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int, mean: torch.Tensor = ..., std: torch.Tensor = ...) -> None:
        super().__init__()
        self.df = df
        self.lookback = lookback
        
        self.mean = mean
        self.std = std
        if mean == ...:
            self.mean = torch.mean(torch.tensor(df.iloc[:, 1:-2].values, dtype=torch.float64), dim=0)
        if std == ...:
            self.std = torch.std(torch.tensor(df.iloc[:, 1:-2].values, dtype=torch.float64), dim=0, unbiased=True)

    def __len__(self) -> int:
        return len(self.df.index) - self.lookback

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.df.index[index]
        j = self.df.index[index + self.lookback - 1]

        data = torch.tensor(self.df.loc[i:j].iloc[:, 1:].values)
        x = data[:, :-2].to(torch.float64)
        move = data[-1, -2].to(torch.float64)
        label = data[-1, -1].to(torch.float64)
        x = (x - self.mean) / self.std
        return x, move, label

def get_daily_dataset(df: pd.DataFrame, look_backward: int, val_start: dt.datetime,
                      test_start: dt.datetime, div_df=None) -> Tuple[DailyDataset, DailyDataset, DailyDataset]:
    all_dates = pd.to_datetime(df["Dates"])

    first_val = df.index[all_dates >= val_start][0]
    first_test = df.index[all_dates >= test_start][0]
    train_df = df.loc[:first_val]
    val_df = df.loc[first_val - look_backward + 1: first_test]
    test_df = df.loc[first_test - look_backward + 1:]
    return DailyDataset(train_df, look_backward, div_df), DailyDataset(val_df, look_backward, div_df), DailyDataset(test_df, look_backward, div_df)

def get_ffd_dataset(df: pd.DataFrame, look_backward: int, val_start: dt.datetime,
                    test_start: dt.datetime) -> Tuple[DatasetFFD, DatasetFFD, DatasetFFD]:
    all_dates = pd.to_datetime(df["Dates"])
    
    first_val = df.index[all_dates >= val_start][0]
    first_test = df.index[all_dates >= test_start][0]
    
    train_df = df.loc[:first_val]
    val_df = df.loc[first_val - look_backward + 1: first_test]
    test_df = df.loc[first_test - look_backward + 1:]

    train_ds = DatasetFFD(train_df, look_backward)
    val_ds = DatasetFFD(val_df, look_backward, train_ds.mean, train_ds.std)
    test_ds = DatasetFFD(test_df, look_backward, train_ds.mean, train_ds.std)
    return train_ds, val_ds, test_ds

# test code
if __name__ == "__main__":
    filename = os.path.join("data", "Day Data with Volatility", "NESZ MK Equity.csv")
    # filename = "NESZ_ffd.csv"
    df = pd.read_csv(filename)
    div_df = pd.read_csv(os.path.join("data", "NESZ dividend.csv"))
    train_ds, val_ds, test_ds = get_daily_dataset(df, 30, dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1), div_df)
    pd.set_option('display.max_rows', 500)
    print(train_ds.df[["Dates", "PX_LAST", "PX_LAST_ADJUSTED"]].tail(100))
    # train_ds, val_ds, test_ds = get_ffd_dataset(df, 30, val_start=dt.datetime(2018, 1, 1), test_start=dt.datetime(2020, 1, 1))
    # print(train_ds[0])
    # print(len(df))
    # print(len(train_ds))
    # print(len(val_ds))
    # print(len(test_ds))

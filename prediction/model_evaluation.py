from typing import List
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import DailyDataset, FEATURES

TRADING_DAYS_IN_YEAR = 261

class ModelEvaluation():
    def __init__(self, daily_returns: List[float], baseline_returns: List[float], conf_matrix: np.ndarray) -> None:
        if len(daily_returns) != len(baseline_returns):
            raise ValueError()
        self.daily_returns = np.array(daily_returns)
        self.baseline_returns = np.array(baseline_returns)
        self.time_length = len(daily_returns)
        self.conf_matrix = conf_matrix

    @property
    def mean_return(self) -> float:
        return np.mean(self.daily_returns)

    @property
    def std_return(self) -> float:
        return np.std(self.daily_returns, ddof=1)

    @property
    def mean_baseline_return(self) -> float:
        return np.mean(self.baseline_returns)

    @property
    def std_baseline_return(self) -> float:
        return np.std(self.baseline_returns, ddof=1)

    @property
    def sharpe(self) -> float:
        return 0 if self.std_return == 0 else self.mean_return / self.std_return * np.sqrt(TRADING_DAYS_IN_YEAR)

    @property
    def baseline_sharpe(self) -> float:
        return 0 if self.std_baseline_return == 0 else self.mean_baseline_return / self.std_baseline_return * np.sqrt(TRADING_DAYS_IN_YEAR)

    @property
    def total_return(self) -> float:
        return np.prod(self.daily_returns + 1) - 1
    
    @property
    def total_baseline_return(self) -> float:
        return np.prod(self.baseline_returns + 1) - 1
    
    @property
    def annualized_return(self) -> float:
        return (1 + self.total_return) ** (TRADING_DAYS_IN_YEAR / self.time_length) - 1

    @property
    def annualized_baseline_return(self) -> float:
        return (1 + self.total_baseline_return) ** (TRADING_DAYS_IN_YEAR / self.time_length) - 1

def evaluate_daily_model(model, val_ds: DailyDataset, device: torch.device) -> ModelEvaluation:
    daily_returns = []
    daily_bnh_returns = []
    conf_matrix = np.zeros((2, 2))

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    for X, _, move in tqdm(val_loader, desc="Eval", leave=False):
        X = X.to(device)
        move = move.numpy()
        y_pred = model(X)
        y_pred = y_pred.cpu()
        X = X.cpu()
        direction = np.where(y_pred >= X[:, -1, 0], 1, -1)
        daily_returns += list(direction * move)
        daily_bnh_returns += list(move)
        true_direction = np.where(move >= 0, 1, -1)
        
        conf_matrix[0, 0] += np.sum((direction == -1) & (true_direction == -1))
        conf_matrix[0, 1] += np.sum((direction == 1) & (true_direction == -1))
        conf_matrix[1, 0] += np.sum((direction == -1) & (true_direction == 1))
        conf_matrix[1, 1] += np.sum((direction == 1) & (true_direction == 1))
    return ModelEvaluation(daily_returns, daily_bnh_returns,conf_matrix)



class EnsembleModel(Module):
    def __init__(self, models) -> None:
        super(EnsembleModel, self).__init__()
        self.models = ModuleList(models)
    
    def forward(self, X, return_std=False) -> torch.Tensor:
        preds = torch.zeros((X.shape[0], len(self.models)))
        for i in range(len(self.models)):
            preds[:, i] = self.models[i](X)
        if return_std:
            return torch.std_mean(preds, dim=1, unbiased=True)
        return torch.mean(preds, dim=1)

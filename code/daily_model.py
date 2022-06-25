
import os
import sys
import datetime as dt
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Module, Linear, GRU, Softmax, ReLU, Tanh, Dropout, BatchNorm1d, ModuleList, MSELoss, init
from torch.utils.data import DataLoader

import pandas as pd

import dataset
from model_evaluation import evaluate_daily_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
class DailyModel(Module):
    def __init__(self, dim_0, dim_1=256, dim_2=512, dropout=0.2, gru_layers=1) -> None:
        super(DailyModel, self).__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2

        self.linear = Linear(dim_0, dim_1)
        self.gru = GRU(dim_1, dim_2, batch_first=True, num_layers=gru_layers)
        self.linear_a = Linear(dim_2, dim_2)
        self.linear_out = Linear(dim_2, 1)

        self.dropout = Dropout(dropout)

        self.relu = ReLU()
        self.tanh = Tanh()
        self.softmax = Softmax(dim=1)

        self.bn_lin = BatchNorm1d(dim_1)
        self.bn_gru = BatchNorm1d(dim_2)
        self.bn_lin_a = BatchNorm1d(dim_2)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]

        # Linear + ReLU
        out = self.linear(x)
        out = self.bn_lin(out.view(-1, self.dim_1)).view(batch_size, -1, self.dim_1)
        out = self.dropout(out)
        out = self.relu(out)

        # GRU
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.bn_gru(out)

        a = self.linear_a(out)
        a = self.bn_lin_a(a)
        a = self.dropout(a)
        a = self.tanh(a)
        a = self.softmax(a)
        out = out * a
        
        # Linear + ReLU
        out = self.linear_out(out)
        return out.view(-1)
'''
class DailyModel(Module):
    def __init__(self, dim_0, dim_1=128, dim_2=256, dropout=0.2, gru_layers=2) -> None:
        super(DailyModel, self).__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2

        self.linear_1 = Linear(dim_0, dim_1)
        self.linear_2 = Linear(dim_1, dim_2)
        self.gru = GRU(dim_2, 2*dim_2, batch_first=True, num_layers=gru_layers, bidirectional = True)
        self.linear_a = Linear(4*dim_2, 4*dim_2)
        self.linear_out = nn.Sequential(Linear(4*dim_2,2*dim_2),
                                        Linear(2*dim_2,1))

        self.dropout = Dropout(dropout)

        self.relu = nn.LeakyReLU()
        self.tanh = Tanh()
        self.softmax = Softmax(dim=1)

        self.bn_lin_1 = BatchNorm1d(dim_1)
        self.bn_lin_2 = BatchNorm1d(dim_2)
        self.bn_gru = BatchNorm1d(4*dim_2)
        self.bn_lin_a = BatchNorm1d(4*dim_2)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]

        # MLPs (2 layers): Linear + ReLU
        out = self.linear_1(x)
        out = self.bn_lin_1(out.view(-1, self.dim_1)).view(batch_size, -1, self.dim_1)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.linear_2(out)
        out = self.bn_lin_2(out.view(-1, self.dim_2)).view(batch_size, -1, self.dim_2)
        out = self.dropout(out)
        out = self.relu(out)


        # GRU
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.bn_gru(out)

        a = self.linear_a(out)
        a = self.bn_lin_a(a)
        a = self.dropout(a)
        a = self.tanh(a)
        a = self.softmax(a)
        out = out * a
        
        # Linear + ReLU
        out = self.linear_out(out)
        return out.view(-1)

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

def weight_init(m) -> None:
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

def train_single_model(train_ds: dataset.DailyDataset, val_ds: dataset.DailyDataset, num_epochs: int=15) -> DailyModel:
    model = DailyModel(len(dataset.FEATURES), dropout=0.1)
    model = model.apply(weight_init).double().to(DEVICE)
    loss_function = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.3)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=64)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=64)

    for e in range(num_epochs):
        model.train()
        running_loss = 0
        count = 0
        for X, y, _ in tqdm(train_loader, desc="Training", leave=False):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            valid_examples = torch.logical_not(
                torch.logical_or(
                    torch.sum(torch.logical_or(torch.isinf(X), torch.isnan(X)), dim=(1,2)) > 0,
                    torch.logical_or(torch.isinf(y), torch.isnan(y))
                )
            )
            X = X[valid_examples]
            y = y[valid_examples]
            if torch.sum(valid_examples) == 0:
                continue

            optimizer.zero_grad()
            y_pred = model(X)

            single_loss = loss_function(y_pred, y)
            if not torch.isnan(single_loss) and not torch.isinf(single_loss):
                single_loss.backward()
                optimizer.step()
                running_loss += single_loss
                count += 1

        scheduler.step()
        if (e+1) % 5 == 0:
            model.eval()
            running_val_loss = 0
            val_count = 0
            for X, y, _ in tqdm(val_loader, desc="Calculating Val loss", leave=False):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                y_pred = model(X)
                single_loss = loss_function(y_pred, y)
                running_val_loss += single_loss
                val_count += 1
                
            print("Epoch {:2d}: Loss = {:4.2f}, Val loss = {:4.2f}".format(e + 1, running_loss / count, running_val_loss / val_count))
        else:
            print("Epoch {:2d}: Loss = {:4.2f}".format(e + 1, running_loss / count))
    
    return model

if __name__ == "__main__":
    start_time = dt.datetime.now()
    print(f"Using {DEVICE}.")
    symbol = "NESZ"
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    filename = os.path.join("data", "Day Data with Volatility", f"{symbol} MK Equity.csv")
    df = pd.read_csv(filename)
    train_ds, val_ds, test_ds = dataset.get_daily_dataset(df, 30, dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1))

    models = []
    
    # Create folder structure if needed
    if not os.path.exists("models"):
        os.mkdir("models")
    daily_model_folder = os.path.join("models", "daily models")
    if not os.path.exists(daily_model_folder):
        os.mkdir(daily_model_folder)
    stock_model_folder = os.path.join(daily_model_folder, symbol)
    if not os.path.exists(stock_model_folder):
        os.mkdir(stock_model_folder)

    # Train 10 individual models
    for i in range(10):
        print("\n--- Model {} ---".format(i+1))
        mod = train_single_model(train_ds, val_ds, num_epochs=15)
        models.append(mod)
        torch.save(mod.state_dict(), os.path.join(stock_model_folder, f"{symbol}_model{i}.pt"))

    # Evaluate ensemble
    print()
    ensemble = EnsembleModel(models)
    train_eval = evaluate_daily_model(ensemble, train_ds, DEVICE)
    val_eval = evaluate_daily_model(ensemble, val_ds, DEVICE)

    print("Sharpe = {:1.2f} ({:1.2f}), Val Sharpe = {:1.2f} ({:1.2f})"
          .format(train_eval.sharpe, train_eval.baseline_sharpe, val_eval.sharpe, val_eval.baseline_sharpe))

    print("Done in {}.".format(dt.datetime.now() - start_time))

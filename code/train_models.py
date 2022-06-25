import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import sys
import datetime as dt

from daily_model import train_single_model
import dataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE}.")

symbols = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]
symbols = symbols[int(sys.argv[1]):int(sys.argv[2])]

for symbol in symbols:
    print(symbols)
    filename = os.path.join("../data", "Day Data with Volatility", f"{symbol} MK Equity.csv")
    df = pd.read_csv(filename)
    train_ds, val_ds, test_ds = dataset.get_daily_dataset(df, 30, dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1))
    
    # Create folder structure if needed
    if not os.path.exists("models"):
        os.mkdir("models")
    daily_model_folder = os.path.join("models", "deeper")
    if not os.path.exists(daily_model_folder):
        os.mkdir(daily_model_folder)
    stock_model_folder = os.path.join(daily_model_folder, symbol)
    if not os.path.exists(stock_model_folder):
        os.mkdir(stock_model_folder)

    # Train 10 individual models
    models = []
    for i in range(10):
        print("\n--- Model {} ---".format(i+1))
        mod = train_single_model(train_ds, val_ds, num_epochs=5)
        models.append(mod.cpu())
        torch.save(mod.state_dict(), os.path.join(stock_model_folder, f"{symbol}_model{i}.pt"))

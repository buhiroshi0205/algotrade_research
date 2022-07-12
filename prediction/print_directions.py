import sys
import os
import datetime as dt

import torch
import pandas as pd
import quantstats as qs
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from daily_model import DailyModel
import dataset

if __name__ == "__main__":
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    STOCKS = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]
    
    for STOCK in STOCKS[int(sys.argv[1]):int(sys.argv[2])]:
        print(STOCK)
        models = [DailyModel(len(dataset.FEATURES), dropout=0.1).double() for _ in range(10)]
        for i in range(10):
            models[i].load_state_dict(torch.load(os.path.join("models", "deeper", STOCK, f"{STOCK}_model{i}.pt")))
            models[i].to(DEVICE)
            models[i].eval()

        filename = os.path.join("data", "Day Data with Volatility", f"{STOCK} MK Equity.csv")
        df = pd.read_csv(filename)
        val_ds = dataset.DailyDataset(df, 30)

        direction_df = pd.DataFrame(index=val_ds.df.loc[val_ds.use_index, "Dates"])
        loader = DataLoader(val_ds, batch_size=64)
        with torch.no_grad():
            for i, (X, _, _) in enumerate(tqdm(loader)):
                # print(f"{i+1}/{len(loader)}")
                all_preds = [m(X.to(DEVICE)).detach() for m in models]
                for k in range(X.shape[0]):
                    preds = [p[k].item() for p in all_preds]
                    direction_df.loc[direction_df.index[i * 64 + k], "AVG"] = np.sign(np.average(preds) - X[k, -1, 0]).item()
                    for j, p in enumerate(preds):
                        direction_df.loc[direction_df.index[i * 64 + k], f"MODEL_{j+1}"] = np.sign(p - X[k, -1, 0]).item()
        
        directions_name = 'deeperdirections'
        if not os.path.exists(directions_name):
            os.mkdir(directions_name)
        direction_df.to_csv(os.path.join(directions_name, f"Directions {STOCK}.csv"))

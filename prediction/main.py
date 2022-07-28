import multiprocessing as mp
import math
import datetime as dt
import time
import os

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import dataset
import train_models

symbols = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]
ensemble_num = 10
experiment_name = 'bestattention_2010split'
processes = 4



def generate_directions(stock, models, device, verbose):
    filename = os.path.join("../data/Day Data with Volatility", "{} MK Equity.csv".format(stock))
    df = pd.read_csv(filename)
    ds = dataset.DailyDataset(df, 30, predict_range=3)
    for m in models:
        m.eval()
        m.to(device)

    direction_df = pd.DataFrame(index=ds.df.loc[ds.use_index, "Dates"])
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    with torch.no_grad():
        for i, (X, y, _) in enumerate(tqdm(loader, desc='generating directions...') if verbose else loader):
            X, y = X.to(device), y.to(device)
            preds = [model(X, y=y, teacher_forcing_rate = 0.95, Gumbel_noise=False, mode='val').squeeze() for model in models]
            all_preds = [pred[:,-1].cpu() for pred in preds]
            X = X.cpu()
            for k in range(X.shape[0]):
                preds = [p[k].item() for p in all_preds]
                direction_df.loc[direction_df.index[i * 64 + k], "AVG"] = np.sign(np.average(preds) - X[k, -1, 0]).item()
                for j, p in enumerate(preds):
                    direction_df.loc[direction_df.index[i * 64 + k], f"MODEL_{j+1}"] = np.sign(p - X[k, -1, 0]).item()
    os.makedirs(f'../data/directions/{experiment_name}', exist_ok=True)
    direction_df.to_csv(f'../data/directions/{experiment_name}/Directions {stock}.csv')


def run(stocks, idx):
    print(f'Process {idx}: {stocks}')
    time.sleep(1)
    pbar = tqdm(total=len(stocks)*ensemble_num, position=idx)
    verbose = 1 if processes == 1 else 0

    for stock in stocks:
        df = pd.read_csv(f'../data/Day Data with Volatility/{stock} MK Equity.csv')
        train_ds, val_ds, test_ds = dataset.get_daily_dataset(df, 30, dt.datetime(2010, 1, 1), dt.datetime(2020, 1, 1), predict_range=3)
        models = []
        for i in range(ensemble_num):
            pbar.set_description(f'{stock}-{i}/{ensemble_num}')
            
            train_config = {
                'model_name': 'attention',
                'model_params': {
                    'input_dim': len(dataset.FEATURES),
                    'encoder_hidden_dim': 128,
                    'decoder_hidden_dim':256,
                    'key_value_size': 128
                },
                'optimizer_name': 'Adam',
                'optimizer_params': {'weight_decay': 0.0001},
                'lr': 0.005,
                'lr_decay': 0.1,
                'epochs': 10,
                'device_name': f'cuda:{idx}'
            }

            model = train_models.train_with_config(train_ds, val_ds, verbose=verbose, **train_config)
            models.append(model)
            #torch.save(model.state_dict(), f'../checkpoints/{experiment_name}/{stock}_model{i}.pt')
            pbar.update()
        generate_directions(stock, models, torch.device(f'cuda:{idx}'), verbose)
    pbar.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.makedirs(f'../checkpoints/{experiment_name}', exist_ok=True)
    if processes == 1:
        run(symbols, 0)
    else:
        npp = int(math.ceil(len(symbols)/processes))
        processes_list = []
        for i in range(processes):
            processes_list.append(mp.Process(target=run, args=(symbols[i*npp:(i+1)*npp], i)))
        for i in range(processes):
            processes_list[i].start()
        for i in range(processes):
            processes_list[i].join()
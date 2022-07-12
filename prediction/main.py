import multiprocessing as mp
import math
import datetime as dt
import time
import os

import pandas as pd
import torch
from tqdm import tqdm

import dataset
import train_models

symbols = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]
ensemble_num = 10
experiment_name = 'temp'
processes = 4


def run(stocks, idx):
    print(f'Process {idx}: {stocks}')
    time.sleep(1)
    pbar = tqdm(total=len(stocks)*ensemble_num, position=idx)
    verbose = 1 if processes == 1 else 0

    for stock in stocks:
        df = pd.read_csv(f'../data/Day Data with Volatility/{stock} MK Equity.csv')
        train_ds, val_ds, test_ds = dataset.get_daily_dataset(df, 30, dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1), predict_range=3)
        for i in range(ensemble_num):
            pbar.set_description(f'{stock}-{i}/{ensemble_num}')

            train_config = {
                'model_name': 'deeper',
                'model_params': {'dim_0': len(dataset.FEATURES)},
                'optimizer_name': 'Adam',
                'optimizer_params': {'weight_decay': 0.0001},
                'epochs': 5,
                'device_name': f'cuda:{idx}'
            }

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
                'lr': 0.001,
                'epochs': 5,
                'device_name': f'cuda:{idx}'
            }

            model = train_models.train_with_config(train_ds, val_ds, verbose=verbose, **train_config)
            torch.save(model.state_dict(), f'../model_checkpoints/{experiment_name}/{stock}_model{i}.pt')
            pbar.update()
    pbar.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.makedirs(f'../model_checkpoints/{experiment_name}', exist_ok=True)
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
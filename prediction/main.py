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


"""
The parameters of the training process NOT RELATED TO training individual models.

:param symbols: the stocks to train predictions on. Each stock gets its independent model, and different stocks don't interact during predictions.
:param ensemble_num: The number of models to train for the ensemble.
:param experiment_name: The name of the experiment. The directions will be saved to the directory '../data/directions/{experiment_name}'.
:param processes: The number of GPUs to use in parallel. Use 1 for testing purposes, as it prints more information and doesn't mess with device stuff.
                  This script will use devices 'cuda:0, 'cuda:1', ..., 'cuda:{processes-1}' so make sure this param is <= the number of GPUs on the machine.
                  If you don't want to use consecutive GPUs (e.g. some gpus are taken), you should specify the available GPUs using the environment variable
                  'CUDA_VISIBLE_DEVICES', and this script will use the first {processes} devices in the visible devices list.
                  
"""
symbols = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]
experiment_name = 'NLL2018'
processes = 4
evaluate_sharpe = False


"""
Generate ensemble directions csv based on trained models.
we wish to PIVOT AWAY from using only directions, and use something more direct like predicted price mean + stdev.
"""
def generate_directions(stock, model, device, verbose):
    filename = os.path.join("../data/Day Data with Volatility", "{} MK Equity.csv".format(stock))
    df = pd.read_csv(filename)
    ds = dataset.DailyDataset(df, 30, predict_range=1, return_scale=True)
    model.eval()
    model.to(device)

    direction_df = pd.DataFrame(index=ds.df.loc[ds.use_index, "Dates"])
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    with torch.no_grad():
        for i, (X, y, _, mean, std) in enumerate(tqdm(loader, desc='generating directions...') if verbose else loader):
            X, y = X.to(device), y.to(device)
            preds = model(X, y=y, teacher_forcing_rate = 0.95, Gumbel_noise=False, mode='val').squeeze().cpu()
            X = X.cpu()
            for k in range(X.shape[0]):
                mu, sigma = preds[k][0], preds[k][1]
                norm_mu = mu * std[k] + mean[k]
                norm_sigma = sigma * std[k]
                norm_x = X[k,-1,0] * std[k] + mean[k]
                direction_df.loc[direction_df.index[i * 64 + k], "MEAN"] = ((norm_mu-norm_x)/norm_x).item()
                direction_df.loc[direction_df.index[i * 64 + k], "STD"] = (norm_sigma/norm_x).item()
    os.makedirs(f'../data/directions/{experiment_name}', exist_ok=True)
    direction_df.to_csv(f'../data/directions/{experiment_name}/Directions {stock}.csv')


# train models and output results!
def run(stocks, idx):
    print(f'Process {idx}: {stocks}')
    time.sleep(1)
    pbar = tqdm(total=len(stocks), position=idx)
    verbose = 1 if processes == 1 else 0

    for stock in stocks:
        df = pd.read_csv(f'../data/Day Data with Volatility/{stock} MK Equity.csv')
        train_ds, val_ds, test_ds = dataset.get_daily_dataset(df, 30, dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1), predict_range=1)
        
        pbar.set_description(f'{stock}')

        # the parameters for training that's RELATED TO training each invidual models.
        # this is manually implemented. REFER TO `train_models.py/train_with_config()` for what options are available.
        train_config = {
            'model_name': 'nll',
            'model_params': {
                'dim_0': len(dataset.FEATURES),
            },
            'optimizer_name': 'Adam',
            'optimizer_params': {'weight_decay': 0.0001},
            'lr': 0.01,
            'gamma': 0.9,
            'epochs': 15,
            'device_name': f'cuda:{idx}' if torch.cuda.is_available() else 'cpu'
        }

        model = train_models.train_with_config(train_ds, val_ds, verbose=verbose, **train_config)
        # you can save the models as .pt weights files as well, but these take up a lot of disk space (last time it was 13G for 20 stocks)
        # so we're not saving it during usual testing runs, and only saving directions csvs.
        # torch.save(model.state_dict(), f'../checkpoints/{experiment_name}/{stock}_model{i}.pt')
        pbar.update()
        
        generate_directions(stock, model, torch.device(f'cuda:{idx}'), verbose)
    pbar.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.makedirs(f'../checkpoints/{experiment_name}', exist_ok=True)

    # multiprocessing is much more complicated, so use processes == 1 for testing unless you're confident.
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
    
    if evaluate_sharpe:
        # import dailyenv in other directory
        import pathlib, sys
        rldir = pathlib.Path(__file__).parent.parent / 'rl'
        if not rldir.exists():
            print('ERROR: Could not find rl module, exiting without sharpe evaluation.')
            exit()
        sys.path.append(str(rldir))
        from dailyenv import DailyTradingEnv
        import quantstats as qs

        # evaluate using baseline rl
        def threshold(obs, n):
            action = np.zeros(len(obs)+1)
            for i in range(len(obs)):
                if obs[i] * ensemble_num >= n:
                    action[i+1] = 1
            if np.sum(action) == 0:
                action[0] = 1
            return action

        # for baseline strategy, loop over n from 0 to {ensemble_num} inclusive
        best_sharpe = -float('inf')
        for n in tqdm(range(ensemble_num+1)):
            env = DailyTradingEnv(symbols, dt.datetime(2018, 1, 1), dt.datetime(2020, 1, 1), experiment_name)
            obs = env.reset()
            done = False
            while not done:
                action = threshold(obs, n=n)
                obs, reward, done, info = env.step(action)
            series = pd.Series(env.balance_record, index=env.dates[:-1])
            if series.std() > 1e-3:
                best_sharpe = max(best_sharpe, qs.stats.sharpe(series))
            else:
                best_sharpe = max(best_sharpe, 0)
        print(f'sharpe = {best_sharpe}')
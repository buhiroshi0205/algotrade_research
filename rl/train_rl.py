import datetime as dt
import multiprocessing as mp
import random
import pprint
import math

from stable_baselines3 import A2C, PPO
import quantstats as qs
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dailyenv import DailyTradingEnv


all_stocks = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]


def evaluate(model, env, metric='sharpe'):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    return qs.stats.sharpe(pd.Series(env.balance_record))


def run(new_hparams={}, n_trials=1, seed=None, name=None, experiment=None, process_idx=0, process_queue=None):
    # default_params
    hparams = {
        'stocks': all_stocks,

        'train_start': 2010,
        'train_end': 2018,
        'train_directions': '2010split',

        'eval_start': 2018,
        'eval_end': 2020,
        'eval_directions': 'olivier',

        'total_ts': int(1e6),
        'eval_ts': int(1e4),

        'gamma': 0,
        'n_steps': 5,
        'lr': 0.0007,
        'ent_coef': 0.0003,
        'depth': 2,
        'width': 64,
    }
    for k, v in new_hparams.items():
        hparams[k] = v
    if name is None:
        name = f'{len(hparams["stocks"])}stock'
        for k, v in new_hparams.items():
            if k not in ['stocks','total_ts','eval_ts']:
                name += f',{k}={v}'
    
    seeds = []
    processes = []
    pipes = []
    for i in range(n_trials):
        newseed = random.randrange(2**31) if seed is None else seed
        seeds.append(newseed)
        
        parent, child = mp.Pipe()
        pipes.append(parent)
        
        p = mp.Process(target=worker, args=(hparams, newseed, child))
        processes.append(p)
        p.start()

    datawriter = SummaryWriter(f'../tensorboard_logs/{name}' if experiment is None else f'../tensorboard_logs/{experiment}/{name}')
    datawriter.add_text('all_hparams', pprint.pformat(hparams), global_step=0)
    datawriter.add_text('seeds', str(seeds), global_step=0)

    train_curve = []
    eval_curve = []
    total_ts = hparams['total_ts']
    eval_ts = hparams['eval_ts']
    curr_ts = 0
    with tqdm(total=total_ts, desc=name, position=process_idx) as pbar:
        while curr_ts < total_ts:

            train_avg = 0
            eval_avg = 0
            valid_results = 0
            for i in range(n_trials):
                a, b = pipes[i].recv()
                if math.isfinite(a) and math.isfinite(b):
                    train_avg += a
                    eval_avg += b
                    valid_results += 1
                else:
                    print(f'NaN/inf error at name={name} seed={seeds[i]} steps={curr_ts}')
            train_avg /= valid_results
            eval_avg /= valid_results
            train_curve.append(train_avg)
            eval_curve.append(eval_avg)

            curr_ts += eval_ts
            datawriter.add_scalar('sharpe/train', train_avg, global_step=curr_ts)
            datawriter.add_scalar('sharpe/eval', eval_avg, global_step=curr_ts)
            pbar.update(eval_ts)

    datawriter.close()
    for p in processes:
        p.join()

    hparams['stocks'] = len(hparams['stocks'])
    hparams['n_trials'] = n_trials
    metrics = {'metrics/max_eval_sharpe': max(eval_curve)}
    hparamwriter = SummaryWriter('../tensorboard_logs' if experiment is None else f'../tensorboard_logs/{experiment}')
    hparamwriter.add_hparams(hparams, metrics, run_name=name)
    hparamwriter.close()

    if process_queue is not None:
        process_queue.put(process_idx)
    return metrics['metrics/max_eval_sharpe']


def worker(hparams, seed, pipe):
    train_env = DailyTradingEnv(hparams['stocks'],
                                dt.datetime(hparams['train_start'], 1, 1),
                                dt.datetime(hparams['train_end'], 1, 1),
                                hparams['train_directions'])
    eval_env = DailyTradingEnv(hparams['stocks'],
                               dt.datetime(hparams['eval_start'], 1, 1), 
                               dt.datetime(hparams['eval_end'], 1, 1), 
                               hparams['eval_directions'])

    total_ts = hparams['total_ts']
    eval_ts = hparams['eval_ts']
    d = hparams['depth']
    w = hparams['width']

    model = A2C('MlpPolicy', train_env, device='cpu',
                learning_rate=hparams['lr'],
                ent_coef=hparams['ent_coef'],
                gamma=hparams['gamma'],
                n_steps=hparams['n_steps'],
                policy_kwargs={'net_arch': [dict(pi=[w]*d, vf=[w]*d)]},
                seed=seed)

    train_curve = []
    eval_curve = []
    
    curr_ts = 0
    while curr_ts < total_ts:
        model.learn(total_timesteps=eval_ts)
        
        train_score = evaluate(model, train_env)
        eval_score = evaluate(model, eval_env)
        pipe.send((train_score, eval_score))
            
        curr_ts += eval_ts


def run_mp(experiment):
    simultaneous_runs = 2

    params_search_list = []
    for depth in [2]:
        for width in [32,64]:
            params = {
                'depth': depth,
                'width': width,
                'n_steps': 20,
                'total_ts': int(2e7),
                'eval_ts': int(5e4),
            }
            params_search_list.append(params)
    print(params_search_list)

    q = mp.Queue()
    for i in range(simultaneous_runs):
        q.put(i)
    for p in tqdm(params_search_list, position=simultaneous_runs, leave=False):
        free_process_idx = q.get()
        mp.Process(target=run, kwargs={
            'new_hparams': p,
            'n_trials': 30,
            'experiment': experiment,
            'process_idx': free_process_idx,
            'process_queue': q
        }).start()
    

def run_once(experiment):
    params = {
        'stocks': all_stocks,
        'train_directions': 'bestattention_2010split',
        'eval_directions': 'bestattention_2018split',
        
        'total_ts': int(1e7),
        'eval_ts': int(5e4),

        'n_steps': 20,
        'depth': 1,
        'width': 32
    }
    run(params, n_trials=20, experiment=experiment)


if __name__ == '__main__':
    pass
    #run_once('best_attention_test')
    #run_mp('20stock_20M_run')

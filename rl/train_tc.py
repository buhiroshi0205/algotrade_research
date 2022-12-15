import datetime as dt
import multiprocessing as mp
import random
import pprint
import math
import time

from stable_baselines3 import A2C, PPO
import quantstats as qs
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from randomenv import RandomEnv


all_stocks = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]


# given a RL model as strategy, evaluates the sharpe ratio of the strategy on env (usually validation set)
def evaluate(model, env, metric='sharpe'):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    if np.std(env.history) < 1e-3:
        return 0
    return qs.stats.sharpe(pd.Series(env.history))

"""
The function that actually trains the RL model based on input parameters.
It is a manager process that spawns many worker processes and averages the results.
Results are logged to tensorboard at the directory ../tensorboard_logs/{experiment}, including training curves, hparams, and seeds.

:param n_trials: the number of processes to to use to run trials. Each process runs one trial and the results are averaged.
:param seed: the seed to use for experiment replication (in which case n_trials should be 1). None = random seed for each trial. Should be kept to None in general.
:param name: the name of the series that is logged to tensorboard. if None then generates automatically, so in general could be kept as None.
:param experiment: the name of the folder to log tensorboard results to. 

Usually each "experiment" consists of multiple "runs", and each "run" consists of multiple "trials".
Each "trial" has the same hparams but different seed, each "run" has different hparams, and each "experiment" has a different purpose that can be achieved through testing different runs.
E.g. experiment = search for best NN architecture, run = attempting depth=2, width=64, trial = one random seed with depth=2, width=64

:param process_idx: deprecated. It is used in properly displaying progress bars with `run_mp()`, but using Optuna for multiprocessing is preferred.
:param process_queue: deprecated. It is used in properly displaying progress bars with `run_mp()`, but using Optuna for multiprocessing is preferred.
"""
def run(new_hparams={}, n_trials=1, seed=None, name=None, experiment=None, process_idx=0, process_queue=None):
    # default_params
    hparams = {
        'stocks': all_stocks,
        'train_split':[(['2000-4-25','2004-7-1'], 'q_1'), (['2000-4-25','2004-7-1'], 'q_2'), (['2000-4-25','2004-7-1'], 'q_3'),
			(['2004-7-1','2009-1-1'], 'q_0'), (['2004-7-1','2009-1-1'], 'q_2'), (['2004-7-1','2009-1-1'], 'q_3'),
			(['2009-1-1','2013-7-1'], 'q_0'), (['2009-1-1','2013-7-1'], 'q_1'), (['2009-1-1','2013-7-1'], 'q_3'),
			(['2013-7-1','2018-1-1'], 'q_0'), (['2013-7-1','2018-1-1'], 'q_1'), (['2013-7-1','2018-1-1'], 'q_2')],
        'val_split':[(['2018-1-1','2020-1-1'], 'olivier')],
        'total_ts': int(5e6),
        'eval_ts': int(1e4),

        'gamma': 0,
        'n_steps': 2048,
        'lr': 1e-3,
        'ent_coef': 1e-3,
        'depth': 1,
        'width': 100,
        'tc':0.00
    }
    # incorporate new hparams
    for k, v in new_hparams.items():
        hparams[k] = v
    # generate a name for the log if not provided
    if name is None:
        name = f'{len(hparams["stocks"])}stock'
    name += f',tc={hparams["tc"]}' # + str(time.time()).split('.')[1][:6]
    
    # for each process, generate a seed, initiate inter-process communication pipe, and start worker.
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

    # write hparams as a string and seeds to tensorboard
    datawriter = SummaryWriter(f'../tensorboard_logs/{name}' if experiment is None else f'../tensorboard_logs/{experiment}/{name}')
    datawriter.add_text('all_hparams', pprint.pformat(hparams), global_step=0)
    datawriter.add_text('seeds', str(seeds), global_step=0)

    # obtain datapoints from each worker
    train_curve = []
    eval_curve = []
    total_ts = hparams['total_ts']
    eval_ts = hparams['eval_ts']
    curr_ts = 0
    with tqdm(total=total_ts, desc=name, position=process_idx) as pbar:
        while curr_ts < total_ts:

            # average the datapoints
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

            # log to tensorboard
            curr_ts += eval_ts
            datawriter.add_scalar('sharpe/train', train_avg, global_step=curr_ts)
            datawriter.add_scalar('sharpe/eval', eval_avg, global_step=curr_ts)
            pbar.update(eval_ts)

    datawriter.close()
    for p in processes:
        p.join()

    # log results to tensorboard, and hparams as values (these need to be done together to get around tensorboard's weird directory structuring)
    hparams['stocks'] = len(hparams['stocks'])
    hparams['n_trials'] = n_trials
    metrics = {'metrics/max_eval_sharpe': max(eval_curve)}
    # hparamwriter = SummaryWriter('../tensorboard_logs' if experiment is None else f'../tensorboard_logs/{experiment}')
    # hparamwriter.add_hparams(hparams, metrics, run_name=name)
    # hparamwriter.close()

    if process_queue is not None:
        process_queue.put(process_idx)
    return metrics['metrics/max_eval_sharpe']


# worker process to actually train the RL algorithm
def worker(hparams, seed, pipe):
    train_env = RandomEnv(hparams['stocks'],hparams['train_split'],tc=0)
    eval_env = RandomEnv(hparams['stocks'],hparams['val_split'],tc=hparams['tc'])

    total_ts = hparams['total_ts']
    eval_ts = hparams['eval_ts']
    d = hparams['depth']
    w = hparams['width']

    model = PPO('MlpPolicy', train_env, device='cpu',
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
        # send train and eval curve datapoints to manager process
        pipe.send((train_score, eval_score))
            
        curr_ts += eval_ts

    # qs.reports.html(stock, download_filename=f'trial_{seed}.html')


# manual multiprocessing code to run multiple "runs" (as defined in the docstring for `run()`) in succession. Can be used for gridsearch.
# DEPRECATED. It is recommended to use Optuna to do multiprocessing.
def run_mp(experiment):
    simultaneous_runs = 4

    params_search_list = []
    for stockidx in range(20):
        params = {
            'stocks': all_stocks[stockidx:stockidx+1]
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
            'n_trials': 20,
            'experiment': experiment,
            'process_idx': free_process_idx,
            'process_queue': q
        }).start()
    

# Run ONE "run" instead of as a bigger experiment.
def run_once(experiment):
    params = {
        'stocks': all_stocks,
        'total_ts': int(2.5e6),
        'eval_ts': int(1e4),
        'tc':0.01
    }
    run(params, n_trials=20, name='no_cost_train' , experiment=experiment)


if __name__ == '__main__':
    run_once('experiment')
    #run_mp('singlestock')

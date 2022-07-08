import datetime as dt
import multiprocessing as mp
import random
import pprint

from stable_baselines3 import A2C, DQN, PPO
import quantstats as qs
import pandas as pd
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


def run(new_hparams={}, n_trials=1, seed=None, name=None):
    # default_params
    hparams = {
        'stocks': all_stocks[:5],

        'train_start': 2010,
        'train_end': 2018,
        'train_directions': '2010split',

        'eval_start': 2018,
        'eval_end': 2020,
        'eval_directions': 'olivier',

        'total_ts': int(1e6),
        'eval_ts': int(1e4),

        'gamma': 0,
        'lr': 0.0007,
        'depth': 2,
        'width': 64,
    }
    for k, v in new_hparams.items():
        hparams[k] = v
    if name is None:
        name = f'{len(hparams["stocks"])}stock'
        for k, v in new_hparams.items():
            name += f',{k}={v}'
    
    seeds = []
    processes = []
    pipes = []
    for i in range(n_trials):
        newseed = random.randrange(2**31) if seed is None else seed
        seeds.append(newseed)
        
        parent, child = mp.Pipe()
        pipes.append(parent)
        
        p = mp.Process(target=worker, args=(hparams, seed, child))
        processes.append(p)
        p.start()

    datawriter = SummaryWriter(f'../tensorboard_logs/{name}')
    datawriter.add_text('all_hparams', pprint.pformat(hparams), global_step=0)
    datawriter.add_text('seeds', str(seeds), global_step=0)

    train_curve = []
    eval_curve = []
    total_ts = hparams['total_ts']
    eval_ts = hparams['eval_ts']
    curr_ts = 0
    with tqdm(total=total_ts, desc=name) as pbar:
        while curr_ts < total_ts:

            train_avg = 0
            eval_avg = 0
            for i in range(n_trials):
                a, b = pipes[i].recv()
                train_avg += a
                eval_avg += b
            train_avg /= n_trials
            eval_avg /= n_trials
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
    hparamwriter = SummaryWriter('../tensorboard_logs')
    hparamwriter.add_hparams(hparams, metrics, run_name=name)
    hparamwriter.close()

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
                gamma=hparams['gamma'],
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


if __name__ == '__main__':
    for width in [4,8,16,32,64]:
        for depth in [1,2,3]:
            params = {
                'depth': depth,
                'width': width
            }
            run(params, n_trials=10)
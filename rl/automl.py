import pickle
import optuna
from train_rl import run

all_stocks = ["AMM", "CIMB", "DIGI", "GAM", "GENM", "GENT", "HLBK", "IOI", "KLK", "MAY", "MISC", "NESZ", "PBK", "PEP", "PETD", "PTG", "RHBBANK", "ROTH", "T", "TNB"]

def objective(trial):
    params = {
        'stocks': all_stocks[:5],
        'total_ts': int(1e6),
        'eval_ts': int(1e4),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 1e-5, 1e-2, log=True),
        'depth': trial.suggest_int('depth', 1, 3),
        'width': trial.suggest_int('width', 1, 128)
    }
    return run(params, n_trials=30, experiment='automl_test')

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    for i in range(100):
        study.optimize(objective, n_trials=1)
        with open(f'study{i}.pkl', 'wb') as f:
            pickle.dump(study, f)
    import pdb
    pdb.set_trace()
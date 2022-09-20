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
    return run(params, n_trials=20, experiment='automl_test')

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    import pdb
    pdb.set_trace()

# to parallelize across different processes, you can use an external storage option.
# different processes on the same machine: sqlite works fine.
# different machines: sqlite doesn't work well over NFS, so use a server-client db system such as mysql.
# refer to Optuna documentation for more information.
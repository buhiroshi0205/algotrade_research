
### July 11
- fixed a bug in averaging of runs where child processes will have the same seed
- reread the 10703 slides on actor critic policy gradient methods and read the entropy part for A(3|2)C paper
- collossal fail on attempt to multiprocess hyperparameter tuning
- will manually gridsearch depth and width as hyperparameters
- remaining things I wanted to do:
  1. [x] rerun all old results and baselines to produce comparable metrics and store in sheets
  2. [x] multiprocess hyperparameter tuning in a correct way
  3. [x] run 20stocks
  4. [x] tune learning rate
  5. [x] run entropy hparam tuning
  6. [ ] run PPO
  7. [ ] run auto-hyperparam tuning like SMAC3, raytune, optuna, etc

### July 12
- Meeting: just continue with the goals from last time
- New multiprocessing code works.
- Running/reproducing old results, getting values I don't remember getting.

### July 13
- Just can't seem to reproduce the old negative results, even though I clearly have a log of "5 stocks actually performing negatively on val set"
- Started running the 20stocks, lr, and ent_coef experiments

### July 14
- Meeting: continue with gridsearch on lr x ent_coef, hparam tuning with 20stocks, and then PPO/other algos
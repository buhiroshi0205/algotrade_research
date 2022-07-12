
### July 11
- fixed a bug in averaging of runs where child processes will have the same seed
- reread the 10703 slides on actor critic policy gradient methods and read the entropy part for A(3|2)C paper
- collossal fail on attempt to multiprocess hyperparameter tuning
- will manually gridsearch depth and width as hyperparameters
- remaining things I wanted to do:
  1. rerun all old results and baselines to produce comparable metrics and store in sheets
  2. multiprocess the heck out of hyperparameter tuning in a correct way
  3. run PPO
  4. run 20stocks
  5. run entropy hparam tuning
  6. run auto-hyperparam tuning like SMAC3, raytune, optuna, etc

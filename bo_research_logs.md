
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
- report more statistics than sharpe ratio on spreadsheet

### July 16
- started running lr x ent_coef gridsearch zoomed near individually best regions so far
- [ ] TODO: keep a separate tensorboard folder of all the BEST runs so far corresponding to the spreadsheet

### July 17
- lr x ent_coef result: original individually tuned params were good: lr=0.0013, ent_coef=0.00031
- started running 20stocks

### July 18
- 20 stock NN architecture search: depth must be shallow. depth3 did worst, 1 was best. Best was 1x32. Peaks happened around 1~2M and started overfitting except 1x64, which peaked 3M and stable after 5M
- Train curve smooth with more capacity = more accuracy when depth=1. depth=3 still bad
- 20 stock lr x ent_coef search seems too noisy, so skeptical about results
- if results are to be believed: lr about 0.007 or <= 0.007 (<0.008 to be precise), and ent_coef <= 0.0003 (huge range for both)
- [x] TODO: categorize tensorboard folders into experiment names for storage

### July 22
- started testing n_steps hyperparameter for some hopes of speedup

### July 23
- n_steps, as expected, shifted the eval/training curve a little. 20 was a good balance.
- started running 20M steps experiment using current best two depth, width, ent_coef, and lr (16 options)

### July 25
- realized that I messed up the previous two experiments, rerunning them

### July 26
- Organized some results out of a mountain of stuff and sent professor an update report

### July 27
- updated direction generating code and generated good directions with best_attention model

### July 28
- using directions from best_attention model to train actually was pretty bad, don't know the reason yet
  - could it be because the model is only good on the eval set but not on the training set?
  - could it be too good so the model learned to fit something too specific?

### July 30
- started running automl with optuna on 5 stocks for testing